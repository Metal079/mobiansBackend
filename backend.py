import os
import io
import base64
import requests
from typing import Optional, List, Dict
import hashlib
import logging
import random
import asyncio
from datetime import datetime
import json
import re
import websockets

import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv
from redis.asyncio import Redis
from redis.backoff import ExponentialBackoff
from redis.retry import Retry
from redis.exceptions import BusyLoadingError, ConnectionError, TimeoutError
from pywebpush import webpush, WebPushException
import numpy as np
import imagehash
import asyncpg

# from pyinstrument import Profiler
# from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
# from starlette.requests import Request
# from starlette.responses import Response

from helper_functions import *

PROFILING = False  # Set this from a settings model

logging.basicConfig(level=logging.ERROR)  # Configure logging

# Run 3 retries with exponential backoff strategy
retry = Retry(ExponentialBackoff(), 3)
load_dotenv()
REDISHOST = os.environ.get("REDISHOST")

DBHOST = os.environ.get("DBHOST")
DBNAME = os.environ.get("DBNAME")
DBUSER = os.environ.get("DBUSER")
DBPASS = os.environ.get("DBPASS")

# Define your connection parameters for PostgreSQL
DATABASE_URL = f"postgresql://{DBUSER}:{DBPASS}@{DBHOST}/{DBNAME}"

API_IP_List = os.environ.get("API_IP_List").split(" ")

VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY")
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY")
VAPID_CLAIMS = os.environ.get("VAPID_CLAIMS")
subscriptions: Dict[str, dict] = {}

app = FastAPI()
global_queue = {}  # This will store the latest queue information
session = None


async def create_db_pool():
    return await asyncpg.create_pool(
        dsn=DATABASE_URL, max_inactive_connection_lifetime=15, max_size=50
    )


@app.on_event("startup")
async def startup_event():
    global session
    global blob_service_client
    global r

    session = aiohttp.ClientSession(trust_env=True)
    # blob_service_client = BlobServiceClient.from_connection_string(
    #     os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    # )
    r = Redis(
        host=REDISHOST,
        port=6379,
        db=0,
        retry=retry,
        retry_on_error=[BusyLoadingError, ConnectionError, TimeoutError],
    )  # , decode_responses=True

    # Create a connection pool
    try:
        app.state.db_pool = await create_db_pool()
    except Exception as e:
        logging.error(f"Failed to create a database pool at startup: {e}")

    for index, ip in enumerate(API_IP_List):
        ws_uri = f"ws://{ip}/ws/queue_length"
        asyncio.create_task(listen_for_queue_updates(ws_uri, index))


@app.on_event("shutdown")
async def shutdown_event():
    await session.close()
    await app.state.db_pool.close()


# profiler = Profiler(interval=0.001, async_mode="enabled")
# profiler_running = False
# last_profile_time = time.time()

# # Middleware to profile requests every 5 minutes
# class PyInstrumentMiddleWare(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
#         global profiler_running
#         global last_profile_time
#         global profiler

#         current_time = time.time()

#         # Start profiler if not already running
#         if not profiler_running:
#             profiler_running = True
#             last_profile_time = current_time
#             profiler.start()

#         response = await call_next(request)

#         # Check if 5 minutes have elapsed to stop the profiler
#         if current_time - last_profile_time > 30:  # 300 seconds = 5 minutes
#             profiler_running = False
#             profiler.stop()
#             profiler.write_html("profile.html")
#             profiler = Profiler(interval=0.001, async_mode="enabled")  # Reset profiler for next cycle
#             last_profile_time = current_time

#         return response


async def get_connection():
    if not hasattr(app.state, "db_pool") or app.state.db_pool is None:
        logging.info("Attempting to create a new database pool.")
        try:
            app.state.db_pool = await create_db_pool()
        except Exception as e:
            logging.error(f"Failed to create a new database pool: {e}")
            app.state.db_pool = (
                None  # Invalidate the pool so it will be recreated next time
            )
            yield None

    if hasattr(app.state, "db_pool") and app.state.db_pool is not None:
        try:
            async with app.state.db_pool.acquire() as connection:
                yield connection
        except Exception as e:
            logging.error(f"Error acquiring connection from pool: {e}")
            app.state.db_pool = (
                None  # Invalidate the pool so it will be recreated next time
            )
            yield None


# app.add_middleware(PyInstrumentMiddleWare)


# Set up the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


class ImageData(BaseModel):
    url: Optional[str] = None
    width: Optional[int]
    height: Optional[int]
    aspectRatio: Optional[str]
    base64: str
    UUID: Optional[str]
    rated: Optional[bool]


class JobData(BaseModel):
    prompt: str
    image: Optional[str] = None
    image_UUID: Optional[str] = None
    mask_image: Optional[str] = None
    color_inpaint: Optional[bool] = None
    control_image: Optional[str] = None
    scheduler: int
    steps: int
    negative_prompt: str
    width: int
    height: int
    guidance_scale: int
    seed: int
    batch_size: int
    strength: Optional[float] = None
    job_type: str
    model: Optional[str] = None
    fast_pass_code: Optional[str] = None
    rating: Optional[bool] = None


class ImageRequestModel(JobData):
    image: Optional[str] = None
    fast_pass_enabled: Optional[bool] = False


async def listen_for_queue_updates(uri, index):
    global global_queue
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                while True:
                    message = await websocket.recv()
                    queue_info = json.loads(message)
                    global_queue[index] = queue_info["queue_length"]
                    # Here, you would handle the received message
                    print(f"Queue Update for {uri}: {global_queue}")
                    # No need to sleep because you're waiting for messages from the server
        except Exception as e:
            print(f"Error connecting to WebSocket at {uri}: {e}")
            global_queue[index] = 9999
            # If the connection fails, wait before trying to reconnect
            await asyncio.sleep(5)


@app.post("/submit_job/")
async def submit_job(
    job_data: JobData,
    background_tasks: BackgroundTasks,
    conn: Optional[asyncpg.Connection] = Depends(get_connection),
):
    # Check if FastPassCode is valid and non-expired
    fast_pass_enabled = False
    if job_data.fast_pass_code:
        try:
            is_valid = await validate_fastpass(
                job_data.fast_pass_code, conn, background_tasks
            )
            if is_valid:
                fast_pass_enabled = True
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid or expired FastPassCode. Please fix/remove the FastPassCode and try again.",
                )
        except Exception as e:
            logging.error(
                "Error occurred while validating FastPassCode (DB might be down))"
            )
            logging.error(str(e))

    # Filter out prompts
    job_data.prompt, job_data.negative_prompt = await promptFilter(job_data)
    job_data.negative_prompt = fortify_default_negative(job_data.negative_prompt)

    API_IP = await chooseAPI()

    # Do img2img filtering if it's an img2img request
    if job_data.job_type == "img2img" or job_data.job_type == "inpainting":
        # Convert base64 string to image to remove alpha channel if needed
        job_data.image = decode_base64_to_image(job_data.image)
        job_data.image = job_data.image.convert("RGBA")

        # Do the same for mask image
        if job_data.mask_image:
            job_data.mask_image = decode_base64_to_image(job_data.mask_image)
            job_data.mask_image = job_data.mask_image.convert("RGBA")

        if job_data.color_inpaint:
            # Check if the mask image has an alpha channel
            if "A" in job_data.mask_image.getbands():
                # Extract the alpha channel and adjust its opacity
                alpha_channel = job_data.mask_image.split()[-1]
                alpha_channel = alpha_channel.point(
                    lambda p: int(p * job_data.strength)
                )

            else:
                # Create a new alpha channel based on the mask image's pixel data
                # alpha_channel = ImageOps.invert(job_data.mask_image.convert("L")).point(lambda p: int(p * 0.7))

                # Give an error
                raise HTTPException(
                    status_code=400, detail="The mask image must have an alpha channel."
                )

            # Create a new mask image with the adjusted alpha channel
            mask_with_alpha = Image.merge(
                "RGBA", job_data.mask_image.split()[:-1] + (alpha_channel,)
            )

            # Overlay the mask image onto the main image using the adjusted alpha channel
            job_data.image.paste(mask_with_alpha, (0, 0), mask=alpha_channel)

            # Convert the PIL Image to a NumPy array
            mask_array = np.array(job_data.mask_image)

            # Identify non-transparent pixels
            not_transparent = mask_array[:, :, 3] > 0  # Alpha channel is not 0

            # Set those pixels to black while maintaining the alpha channel
            mask_array[not_transparent, :3] = 0  # Set R, G, B to 0

            # Convert the NumPy array back to a PIL image
            job_data.mask_image = Image.fromarray(mask_array, "RGBA")

        # Remove alpha channel from image and mask image
        job_data.image = remove_alpha_channel(job_data.image)
        if job_data.mask_image:
            job_data.mask_image = remove_alpha_channel(job_data.mask_image)

        # Save resized image to a BytesIO object
        buffer = io.BytesIO()
        job_data.image.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode BytesIO object to base64
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        job_data.image = encoded_image

        if job_data.mask_image:
            # Save resized image to a BytesIO object
            buffer = io.BytesIO()
            job_data.mask_image.save(buffer, format="PNG")
            buffer.seek(0)

            # Encode BytesIO object to base64
            encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            job_data.mask_image = encoded_image

    # Create an instance of ImageRequestModel
    image_request_data = ImageRequestModel(
        **job_data.dict(), fast_pass_enabled=fast_pass_enabled
    )

    # Try using the requested API, if it fails, use the other one
    returned_data = None
    try:
        async with session.post(
            f"http://{API_IP}/submit_job/", json=image_request_data.dict()
        ) as resp:
            returned_data = await resp.json()
    except:
        API_IP = await chooseAPI()  # Ensure chooseAPI is also async
        async with session.post(
            f"http://{API_IP}/submit_job/", json=image_request_data.dict()
        ) as resp:
            returned_data = await resp.json()

    # attempts = 0
    # while response.status != 200 and attempts < 3:
    #     API_IP = await chooseAPI()  # Ensure chooseAPI is also async
    #     print(f"got error: {response.status} for submit_job, api: {API_IP}")
    #     attempts += 1
    #     async with session.post(f"http://{API_IP}/submit_job/", json=image_request_data.dict()) as resp:
    #         returned_data = await resp.json()
    #     await asyncio.sleep(1)

    # Get index of API_IP in API_IP_List
    for i in range(len(API_IP_List)):
        if API_IP_List[i] == API_IP:
            returned_data["API_IP"] = i

    if returned_data == None:
        raise HTTPException(
            status_code=500, detail="Error occurred while submitting job"
        )

    return JSONResponse(content=returned_data)


def decode_base64_to_image(base64_str):
    # Convert base64 string to image
    try:
        image = Image.open(io.BytesIO(base64.b64decode(base64_str.split(",", 1)[1])))
    except:
        image = Image.open(io.BytesIO(base64.b64decode(base64_str.split(",", 1)[0])))

    # Resize image if needed
    tempAspectRatio = image.width / image.height
    if tempAspectRatio < 0.8:
        image = image.resize((512, 768))
    elif tempAspectRatio > 1.2:
        image = image.resize((768, 512))
    else:
        image = image.resize((512, 512))

    return image


async def increment_fastpass_use_count(fast_pass_code: str, conn: asyncpg.Connection):
    await conn.execute(
        """
        UPDATE fastpass
        SET uses = uses + 1
        WHERE fastpass_code = $1
        """,
        fast_pass_code,
    )


async def validate_fastpass(
    fast_pass_code: str, conn: asyncpg.Connection, background_tasks: BackgroundTasks
) -> bool:
    row = await conn.fetchrow(
        """
        SELECT expiration_date
        FROM fastpass
        WHERE fastpass_code = $1
        """,
        fast_pass_code,
    )

    if not row:
        return False

    expiration_date = row["expiration_date"]
    if expiration_date is None:
        background_tasks.add_task(increment_fastpass_use_count, fast_pass_code, conn)
        return True
    elif expiration_date < datetime.now():
        return False

    background_tasks.add_task(increment_fastpass_use_count, fast_pass_code, conn)
    return True


class GetJobData(BaseModel):
    job_id: str
    API_IP: int


class JobRetryInfo(BaseModel):
    job_id: str
    indexes: List[int]


async def call_get_job(job_id: str, API_IP: str):
    async with session.get(
        url=f"http://{API_IP}/get_job/{job_id}", ssl=False
    ) as response:
        return await response.json()


async def insert_image_hashes(
    image_hashes, metadata, job_data, conn: asyncpg.Connection = Depends(get_connection)
):
    async with conn.transaction():  # Start a transaction
        insert_query = """
            INSERT INTO ImageHashes (hash, prompt, negative_prompt, seed, cfg, model, created_date)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        values = [
            (
                image_hashes[i],
                metadata.prompt,
                metadata.negative_prompt,
                metadata.seed,
                metadata.guidance_scale,
                "Sonic DiffusionV4",
                datetime.now(),
            )
            for i in range(4)
        ]

        # Use executemany to insert multiple records
        await conn.executemany(insert_query, values)


async def process_images_and_store_hashes(image_results, metadata, job_data, conn):
    image_hashes = []
    for i in range(4):
        image = Image.open(io.BytesIO(image_results[2 * i]))
        image_hash = imagehash.phash(image, 16)
        image_hashes.append(str(image_hash))

    try:
        await insert_image_hashes(image_hashes, metadata, job_data, conn)
    except Exception as e:
        logging.error(
            f"Error occurred while inserting image hash info into DB, JOB: {job_data.job_id}"
        )
        logging.error(str(e))


@app.post("/get_job/")
async def get_job(
    job_data: GetJobData,
    background_tasks: BackgroundTasks,
    conn: Optional[asyncpg.Connection] = Depends(get_connection),
):
    MAX_RETRIES = 3
    MIN_DELAY = 1
    MAX_DELAY = 60

    response = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await call_get_job(job_data.job_id, API_IP_List[job_data.API_IP])
            # response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
            break  # success, no need for more retries
        except Exception as e:
            if attempt == MAX_RETRIES - 1:  # if this was the last attempt
                logging.error(
                    f"Max retries exceeded when making GET request, JOB: {job_data.job_id}"
                )
                return JSONResponse(
                    content={"message": "Error occurred while making GET request"},
                    status_code=500,
                )
            else:
                # Calculate next sleep time
                sleep_time = MIN_DELAY * (2**attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, MAX_DELAY)

                logging.error(e)
                logging.error(
                    f"Exception occurred when making GET request, JOB: {job_data.job_id}. Retrying in {sleep_time} seconds..."
                )
                await asyncio.sleep(sleep_time)

    try:
        if response["status"] == "completed":
            async with r.pipeline() as pipe:
                # Fetch images from Redis
                finished_response = {"status": "completed", "result": []}
                metadata_json = await r.get(
                    f"job:{job_data.job_id}:metadata"
                )  # Changed to async
                metadata = JobData.parse_raw(metadata_json)

                # First pass: Identify corrupted images
                image_keys = [f"job:{job_data.job_id}:image:{i}" for i in range(4)]
                checksum_keys = [
                    f"job:{job_data.job_id}:image:{i}:checksum" for i in range(4)
                ]

                # Fetch images and checksums
                for i in range(4):
                    pipe.get(image_keys[i])
                    pipe.get(checksum_keys[i])
                results = await pipe.execute()

                # Check for corrupted images
                corrupted_indexes = []
                for i in range(4):
                    image_bytes = results[2 * i]
                    fetched_checksum = results[2 * i + 1]

                    if image_bytes is not None and fetched_checksum is not None:
                        computed_checksum = hashlib.sha256(image_bytes).hexdigest()
                        if fetched_checksum.decode() != computed_checksum:
                            corrupted_indexes.append(i)

                # If there are corrupted images, resend them
                if corrupted_indexes:
                    logging.info(
                        f"Corrupted images detected for job {job_data.job_id}, resending corrupted images"
                    )
                    retry_info = JobRetryInfo(
                        job_id=job_data.job_id, indexes=corrupted_indexes
                    )
                    requests.get(
                        url=f"http://{API_IP_List[job_data.API_IP]}/resend_images/{job_data.job_id}",
                        json=retry_info.dict(),
                    )

                # Second pass: Fetch images, re-attempting if necessary
                attempts = 0
                while attempts < 2:
                    # Fetch images and checksums if corrupted images were detected
                    if corrupted_indexes:
                        for i in range(4):
                            pipe.get(image_keys[i])
                            pipe.get(checksum_keys[i])
                        results = await pipe.execute()

                    # Watermark images or error if corrupted images are still detected
                    for i in range(4):
                        image_bytes = results[2 * i]
                        fetched_checksum = results[2 * i + 1]

                        if image_bytes is not None and fetched_checksum is not None:
                            computed_checksum = hashlib.sha256(image_bytes).hexdigest()
                            if fetched_checksum.decode() == computed_checksum:
                                image = Image.open(io.BytesIO(image_bytes))

                                # Add watermark and metadata
                                watermarked_image = add_watermark(image.convert("RGB"))
                                watermarked_image_base64 = add_image_metadata(
                                    watermarked_image, metadata
                                )
                                finished_response["result"].append(
                                    watermarked_image_base64
                                )
                        else:
                            logging.error(
                                f"Corrupted image STILL detected for job {job_data.job_id}"
                            )

                    if len(finished_response["result"]) == 4:
                        break
                    else:
                        attempts += 1

                # Generate hashes for each image and store them in DB along with image info
                # Pass the results for images and other necessary data to the background task
                background_tasks.add_task(
                    process_images_and_store_hashes, results, metadata, job_data, conn
                )

                return JSONResponse(content=finished_response)

    except Exception as e:
        logging.error(f"response: {response}")
        logging.error(f"Exception: {e}")
        logging.error(f"Exception happened on get_job")

    return JSONResponse(content=response)


async def call_api(api, session):
    try:
        async with session.get(
            url=f"{api}/get_queue_length/", timeout=5, ssl=False
        ) as response:
            return await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"API {api} is down. Error: {str(e)}")
        return {"queue_length": 9999, "api": api}


async def get_queue_length_via_websocket(api_url):
    try:
        # Replace "ws" with "wss" for secure WebSockets over TLS/SSL
        async with websockets.connect(api_url) as websocket:
            # You can send a message if needed, for example, to authenticate
            # await websocket.send('some message')

            # Wait for the server to send a message and parse it as JSON
            message = await websocket.recv()
            return json.loads(message)  # Assuming the server sends JSON
    except Exception as e:
        print(f"Error connecting to WebSocket at {api_url}: {e}")
        return {"queue_length": 9999, "api": api_url}


# Get the queue length of each API and choose the one with the shortest queue
async def chooseAPI():
    lowest_queue = {None: 9999}
    current_key = None
    for api in global_queue:
        if global_queue[api] < lowest_queue[current_key]:
            lowest_queue = {api: global_queue[api]}
            current_key = api

    return API_IP_List[list(lowest_queue.keys())[0]]


def enhanced_filter(prompt, pattern, replacement):
    # Replace spaces with \W+ to match any non-word characters between the words
    pattern = re.sub(r" ", r"\\W+", pattern)
    return re.sub(r"(?i)\b" + pattern + r"\b", replacement, prompt)


async def promptFilter(data):
    prompt = data.prompt
    negative_prompt = data.negative_prompt

    # Common character mispellings
    corrections = {
        "cream the rabbit": [
            "creem the rabbit",
            "creme the rabbit",
            "cram the rabbit",
            "crem the rabbit",
            "craem the rabbit",
            "creamm the rabbit",
            "crema the rabbit",
            "creamie the rabbit",
        ],
        "rosy the rascal": [
            "rosey the rascal",
            "rosie the rascal",
            "rosi the rascal",
            "rosyy the rascal",
        ],
        "charmy the bee": [
            "charmi the bee",
            "charmyy the bee",
            "charmie the bee",
            "charme the bee",
        ],
        "sage": ["sagee"],
        "marine the raccoon": [
            "marin the raccoon",
            "marina the racoon",
            "marinee the raccoon",
        ],
    }

    # Update any above misspellings in the prompt with correct spelling
    for correct, misspellings in corrections.items():
        for misspelling in misspellings:
            prompt = enhanced_filter(prompt, re.escape(misspelling), correct)

    # If above is in prompt we grab artist list from DB and remove them if they were in the prompt
    artist_list = []
    # try:
    #     async with aioodbc.connect(dsn=dsn) as conn:
    #         async with conn.cursor() as cursor:
    #             await cursor.execute("SELECT Phrase FROM FilteredPhrases")
    #             rows = await cursor.fetchall()
    #             for row in rows:
    #                 artist_list.append(row[0])

    #     # Check and remove any filtered phrases from the prompt
    #     for phrase in artist_list:
    #         prompt = prompt.replace(phrase, "")
    # except Exception as e:
    #     print(f"Database error encountered: {e}")

    character_list = [
        "cream the rabbit",
        "rosy the rascal",
        "sage",
        "maria robotnik",
        "marine the raccoon",
        "charmy the bee",
    ]

    censored_tags = [
        "breast",
        "nipple",
        "pussy",
        "nsfw",
        "nudity",
        "naked",
        "loli",
        "nude",
        "ass",
        "rape",
        "sex",
        "boob",
        "sex",
        "busty",
        "tits",
        "thigh",
        "thick",
        "underwear",
        "panties",
        "upskirt",
        "cum",
        "dick",
        "topless",
        "penis",
        "blowjob",
        "ahegao",
        "nude",
        "hips",
        "areola",
        "pantyhose",
        "creampie",
        "position",
        "wet",
        "autocunnilingus",
        "squirting",
        "straddling",
        "girl on top",
        "reverse cowgirl",
        "feet",
        "toes",
        "footjob",
        "vagina",
        "clitoris",
        "furry with non-furry",
        "spread legs",
        "navel",
        "bimbo",
        "fishnet",
        "hourglass figure",
        "slut",
        "interspecies",
        "hetero",
        "tongue",
        "saliva" "anal",
        "penetration",
        "anus",
        "erection",
        "masterbation",
        "butt",
        "thighhighs",
        "lube",
        "lingerie",
        "bent over",
        "doggystyle",
        "sexy",
        "Areolae",
        "exhibitionism",
        "bottomless",
        "shirt lift",
        "no bra",
        "curvy",
        "groin",
        "clothes lift",
        "stomach",
        "spreading legs",
        "hentai",
        "penetrated",
        "masturbating",
        "masturbate",
        "horny",
        "orgasm",
        "fingering",
        "voluptuous",
        "sperm",
        "handjob",
        "gangbang",
        "ejaculation",
        "uncensored",
        "Lifting Skirt",
        "mooning",
        "hindquarters",
        "presenting",
        "porn",
        "latex",
        "fellatio",
        "oral",
        "open legs",
        "spread wide",
        "fucked",
        "fucking",
        "g-string",
        "seductive gaze",
        "dress lift",
        "cleavage",
        "provocative",
        "venus body",
        "revealing clothes",
        "oppai",
        "milf",
        "wardrobe malfunction",
        "clothing aside",
        "micro bikini",
        "thong",
        "gstring",
        "mating",
        "fuck",
        "tentacle",
        "moan",
        "facial",
        "swimsuit to the side",
        "ripped dress",
        "giant chest",
        "Titjob",
        "lesbian",
        "french kiss",
        "furry with furry",
        "clit",
        "Vulva",
        "lust",
        "Libido",
        "Garter",
        "striptease",
        "cock",
        "plump",
        "thicc",
        "scissoring",
        "skimpy",
        "Anal",
        "curvaceous",
        "gaping",
        "string bikini",
        "cunnilingus",
        "Panty",
        "cameltoe",
        "dominatrix",
        "Corset",
        "lewd",
        "Explicit",
        "futanari",
        "foreskin",
        "urethra",
        "skirt lift",
        "bedroom eyes",
        "pregnant",
        "nudist",
        "undressing",
        "black bra",
        "aroused",
        "yuri",
        "d-cup",
        "skindentation",
        "seductive",
        "booty",
        "big melons",
        "testicles",
        "bodily fluid",
        "semen",
        "erect",
        "twerking",
        "lactating",
        "stockings",
        "cowgirl posicion",
        "vaginia",
        "masturbation",
        "pants pull",
        "clothes pull",
        "genital",
        "ming",
        "nudly",
        "breeding",
        "orgy",
        "pinned down",
        "thrusting",
        "cervical",
        "ecstasy",
    ]

    # If character is in prompt, filter out censored tags from prompt
    if any(character in prompt.lower() for character in character_list):
        for tag in censored_tags:
            prompt = prompt.lower().replace(tag.lower(), "")

        # If prompt is changed remove the prompt "blush" from prompt
        if prompt != data.prompt.lower():
            prompt = prompt.replace("blush", "")

        negative_prompt = (
            "(cleavage), navel, 3d, blush, sweat, ((underwear)), (bikini), (nipples), sex, (breasts), nude, "
            + negative_prompt
        )
        logging.error(prompt)

    return prompt, negative_prompt


def fortify_default_negative(negative_prompt):
    if "nsfw" in negative_prompt.lower() and "nipples" not in negative_prompt.lower():
        return "nipples, pussy, breasts, " + negative_prompt
    else:
        return negative_prompt


def filter_seed(data):
    seed = data["seed"]
    try:
        if not isinstance(data["data"]["seed"], int):
            seed = 999999
        elif data["data"]["seed"] > 999999 | data["data"]["seed"] < 1:
            seed = 999999
    except:
        seed = 1

    return seed


# async def upload_blob(blob_service_client, container_name, blob_name, data):
#     blob_client = blob_service_client.get_blob_client(container_name, blob_name)

#     # Check if the blob exists
#     if await blob_client.exists():
#         try:
#             # Delete the existing blob
#             await blob_client.delete_blob()
#         except Exception as e:
#             print(f"An error occurred while deleting blob {blob_name}: {e}")
#     else:
#         try:
#             # Upload the new blob
#             await blob_client.upload_blob(data)
#             return blob_client.url
#         except Exception as e:
#             print(f"An error occurred while uploading blob {blob_name}: {e}")


# async def delete_and_insert_image_metadata(image_details, blob_url, dsn, rating, uuid):
#     async with aioodbc.connect(dsn=dsn) as conn:
#         async with conn.cursor() as cursor:
#             # Check if the UUID already exists
#             check_query = "SELECT * FROM UserRatings WHERE FileName = ?"
#             await cursor.execute(check_query, (uuid,))
#             existing_record = await cursor.fetchone()

#             if existing_record:
#                 # If the UUID exists, delete the existing record
#                 delete_query = "DELETE FROM UserRatings WHERE FileName = ?"
#                 await cursor.execute(delete_query, (uuid,))
#                 return "deleted"

#             else:
#                 # Insert a new record
#                 insert_query = """
#                     INSERT INTO UserRatings (Prompt, NegativePrompt, Seed, CFG, FileName, RateDate, UserRating, JobType, AzureBlobURL)
#                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
#                 """
#                 file_name = uuid
#                 rate_date = datetime.now()
#                 await cursor.execute(
#                     insert_query,
#                     (
#                         image_details["prompt"],
#                         image_details["negative_prompt"],
#                         image_details["seed"],
#                         image_details["cfg"],
#                         file_name,
#                         rate_date,
#                         rating,
#                         image_details["job_type"],
#                         blob_url,
#                     ),
#                 )

#                 await conn.commit()
#                 return "inserted"


@app.post("/rate_image/")
async def rate_image(job_data: JobData):
    # # Decode base64 image and convert it to bytes
    # try:
    #     image_bytes = base64.b64decode(job_data.image.split(",", 1)[0])
    # except:
    #     image_bytes = base64.b64decode(job_data.image.split(",", 1)[1])

    # image = Image.open(BytesIO(image_bytes))

    # info = image.info

    # # Get image metadata
    # image_details = {}
    # for key, value in info.items():
    #     image_details[key] = value

    # # Generate a random filename
    # image_name = job_data.image_UUID

    # # Upload to Azure Blob Storage
    # container_name = "mobiansratings"
    # blob_url = await upload_blob(
    #     blob_service_client, container_name, image_name, image_bytes
    # )

    # # Insert image metadata into database
    # db_status = await delete_and_insert_image_metadata(
    #     image_details, blob_url, dsn, rating=job_data.rating, uuid=image_name
    # )

    return JSONResponse({"message": f"we no longer use this"})


class Subscription(BaseModel):
    userId: str
    endpoint: str
    expirationTime: Optional[str]
    keys: dict


@app.post("/subscribe")
async def subscribe(subscription: Subscription):
    user_id = subscription.userId
    subscriptions[user_id] = subscription.dict()
    return {"status": "subscribed"}


@app.get("/send_notification/{user_id}")
async def send_notification(user_id: str):  # Change the type to str
    # Retrieve the subscription object for the user
    subscription = subscriptions.get(user_id)
    if not subscription:
        return {"status": "failed", "detail": "User not subscribed"}

    try:
        payload = {
            "notification": {
                "title": "Your image is ready!",
                "body": "Click to view your image.",
                # "icon": "icon.png",
                "vibrate": [100, 50, 100],
                "data": {"url": "https://mobians.ai/"},
            }
        }
        webpush(
            subscription_info=subscription,
            data=json.dumps(payload),
            vapid_private_key=VAPID_PRIVATE_KEY,
            vapid_claims={"sub": "mailto:your_email@example.com"},
        )
    except WebPushException as e:
        print("Failed to send notification:", repr(e))
        return {"status": "failed", "detail": repr(e)}

    return {"status": "sent"}
