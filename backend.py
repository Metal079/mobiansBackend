import os
import io
import base64
from typing import Optional, List, Dict
import hashlib
import logging
import random
import asyncio
from datetime import datetime
import json
import re
import websockets
import time
import requests

from contextlib import asynccontextmanager
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
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
import psycopg

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
DSN = f"host={DBHOST} dbname='{DBNAME}' user={DBUSER} password={DBPASS}"

API_IP_List = os.environ.get("API_IP_List").split(" ")

VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY")
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY")
VAPID_CLAIMS = os.environ.get("VAPID_CLAIMS")
subscriptions: Dict[str, dict] = {}

app = FastAPI()
global_queue = {}  # This will store the latest queue information
fastpass_cache = {}  # In-memory cache for FastPass data
session = None


@app.on_event("startup")
async def startup_event():
    global session
    # global blob_service_client
    global r

    session = aiohttp.ClientSession(trust_env=True)
    r = Redis(
        host=REDISHOST,
        port=6379,
        db=0,
        retry=retry,
        retry_on_error=[BusyLoadingError, ConnectionError, TimeoutError],
    )  # , decode_responses=True

    asyncio.create_task(refresh_fastpass_cache())
    for index, ip in enumerate(API_IP_List):
        ws_uri = f"ws://{ip}/ws/queue_length"
        asyncio.create_task(listen_for_queue_updates(ws_uri, index))


@app.on_event("shutdown")
async def shutdown_event():
    await session.close()
    # await app.state.db_pool.close()


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
                    # print(f"Queue Update for {uri}: {global_queue}")
                    # No need to sleep because you're waiting for messages from the server
        except Exception as e:
            # print(f"Error connecting to WebSocket at {uri}: {e}")
            global_queue[index] = 9999
            # If the connection fails, wait before trying to reconnect
            await asyncio.sleep(3)


async def refresh_fastpass_cache():
    while True:
        async with await psycopg.AsyncConnection.connect(DSN) as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(
                    "SELECT fastpass_code, expiration_date FROM fastpass"
                )
                rows = await acur.fetchall()

                for row in rows:
                    fastpass_code, expiration_date = row
                    fastpass_cache[fastpass_code] = expiration_date

        await asyncio.sleep(300)  # Refresh every 5 minutes (300 seconds)


@app.post("/submit_job/")
async def submit_job(
    job_data: JobData,
    background_tasks: BackgroundTasks,
):
    # Check if FastPassCode is valid and non-expired
    fast_pass_enabled = False
    if job_data.fast_pass_code:
        try:
            is_valid = await validate_fastpass(
                job_data.fast_pass_code, background_tasks
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
    job_data.negative_prompt = await fortify_default_negative(job_data.negative_prompt)

    API_IP = await chooseAPI()

    # Do img2img filtering if it's an img2img request
    if (
        job_data.job_type == "img2img"
        or job_data.job_type == "inpainting"
        or job_data.job_type == "upscale"
    ):

        def upscale(image):
            width, height = image.size
            new_width = int(width * 2)
            new_height = int(height * 2)
            return image.resize((new_width, new_height), Image.LANCZOS)

        # Convert base64 string to image to remove alpha channel if needed
        job_data.image = decode_base64_to_image(job_data.image)
        # Upscale if job_type is upscale
        if job_data.job_type == "upscale":
            job_data.image = upscale(job_data.image)

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
        job_data.image = await remove_alpha_channel(job_data.image)
        if job_data.mask_image:
            job_data.mask_image = await remove_alpha_channel(job_data.mask_image)

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


async def increment_fastpass_use_count(fast_pass_code: str):
    # Time the function
    start_time = time.time()
    logging.info("Incrementing FastPassCode use count")

    async with await psycopg.AsyncConnection.connect(DSN) as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE fastpass
                SET uses = uses + 1
                WHERE fastpass_code = %s
                """,
                (fast_pass_code,),
            )

    # Time the function
    end_time = time.time()
    logging.info("Time elapsed: " + str(end_time - start_time))


async def validate_fastpass(
    fast_pass_code: str, background_tasks: BackgroundTasks
) -> bool:
    if fast_pass_code in fastpass_cache:
        expiration_date = fastpass_cache[fast_pass_code]
        if expiration_date is None or expiration_date >= datetime.now():
            background_tasks.add_task(increment_fastpass_use_count, fast_pass_code)
            return True
        else:
            return False
    else:
        # FastPass data not found in cache, query the database
        async with await psycopg.AsyncConnection.connect(DSN) as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(
                    """
                    SELECT expiration_date
                    FROM fastpass
                    WHERE fastpass_code = %s
                    """,
                    (fast_pass_code,),
                )

                row = await acur.fetchone()

                if not row:
                    return False

                expiration_date = row[0]
                fastpass_cache[fast_pass_code] = expiration_date  # Store in cache

                if expiration_date is None or expiration_date >= datetime.now():
                    background_tasks.add_task(
                        increment_fastpass_use_count, fast_pass_code
                    )
                    return True
                else:
                    return False


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


async def insert_image_hashes(image_hashes, metadata, job_data):
    logging.info("Inserting image hashes")

    insert_query = """
        INSERT INTO hashes (hash, prompt, negative_prompt, seed, cfg, model, created_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
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

    async with await psycopg.AsyncConnection.connect(DSN) as aconn:
        async with aconn.cursor() as acur:
            # Use executemany to insert multiple records
            await acur.executemany(insert_query, values)
            await aconn.commit()  # Commit the transaction


async def twos_complement(hexstr, bits):
    value = int(hexstr, 16)  # convert hexadecimal to integer

    # convert from unsigned number to signed number with "bits" bits
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


async def process_images_and_store_hashes(image_results, metadata, job_data):
    image_hashes = []
    for i in range(4):
        image = Image.open(io.BytesIO(image_results[2 * i]))
        image_hash = imagehash.average_hash(image, 8)
        image_hash = await twos_complement(str(image_hash), 64)
        image_hashes.append(image_hash)

    try:
        await insert_image_hashes(image_hashes, metadata, job_data)
    except Exception as e:
        logging.error(
            f"Error occurred while inserting image hash info into DB, JOB: {job_data.job_id}"
        )
        logging.error(str(e))


@app.post("/get_job/")
async def get_job(
    job_data: GetJobData,
    background_tasks: BackgroundTasks,
):
    MAX_RETRIES = 2
    MIN_DELAY = 1
    MAX_DELAY = 3

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

    error_flag = False
    try:
        if response["status"] == "completed":
            return await retrieve_finished_job(job_data, background_tasks)

    except Exception as e:
        logging.error(f"response: {response}")
        logging.error(f"Exception: {e}")
        logging.error(f"Exception happened on get_job")
        error_flag = True

    #  Try to get the images again if theres an exception
    if error_flag:
        try:
            return await retrieve_finished_job(job_data, background_tasks)
        except Exception as e:
            logging.error(f"Exception: {e}")
            logging.error(f"Exception happened on retrieve_finished_job")

    return JSONResponse(content=response)


async def retrieve_finished_job(
    job_data: GetJobData,
    background_tasks: BackgroundTasks,
):
    async with r.pipeline() as pipe:
        # Fetch images from Redis
        finished_response = {"status": "completed", "result": []}
        metadata_json = await r.get(
            f"job:{job_data.job_id}:metadata"
        )  # Changed to async
        try:
            metadata = JobData.model_validate_json(metadata_json)
        except Exception as e:
            corrupted_indexes = [0, 1, 2, 3]
            retry_info = JobRetryInfo(job_id=job_data.job_id, indexes=corrupted_indexes)
            resp = requests.get(
                url=f"http://{API_IP_List[job_data.API_IP]}/resend_images/{job_data.job_id}",
                json=retry_info.dict(),
            )
            logging.info(resp.json())
            logging.error(f"Error occurred while fetching metadata: {e}")
            # shiw metadata_json
            logging.error(f"metadata_json: {metadata_json}")
            # return JSONResponse(
            #     status_code=500,
            #     content={"message": "Error occurred while fetching metadata"},
            # )
            # ignore it
            metadata = None

        # First pass: Identify corrupted images
        image_keys = [f"job:{job_data.job_id}:image:{i}" for i in range(4)]
        checksum_keys = [f"job:{job_data.job_id}:image:{i}:checksum" for i in range(4)]

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
            retry_info = JobRetryInfo(job_id=job_data.job_id, indexes=corrupted_indexes)
            # requests.get(
            #     url=f"http://{API_IP_List[job_data.API_IP]}/resend_images/{job_data.job_id}",
            #     json=retry_info.dict(),
            # )
            url = (
                f"http://{API_IP_List[job_data.API_IP]}/resend_images/{job_data.job_id}"
            )
            headers = {"Content-Type": "application/json"}

            async with session.get(url, headers=retry_info.model_dump()) as user_resp:
                if user_resp.status != 200:
                    raise HTTPException(
                        status_code=user_resp.status,
                        detail="E",
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
                        watermarked_image = await add_watermark(image.convert("RGB"))
                        watermarked_image_base64 = await add_image_metadata(
                            watermarked_image, metadata
                        )
                        finished_response["result"].append(watermarked_image_base64)
                else:
                    logging.error(
                        f"Corrupted image STILL detected for job {job_data.job_id}"
                    )
                    logging.error(f"Corrupted indexes: {corrupted_indexes}")
                    logging.error(f"Results: {results}")

            if len(finished_response["result"]) == 4:
                break
            else:
                attempts += 1

        if len(finished_response["result"]) < 4 and attempts >= 2:
            # Not all images could be processed successfully
            error_response = {
                "status": "error",
                "message": "Unable to process all images due to corruption.",
                "processed_images_count": len(finished_response["result"]),
                "total_images_expected": 4,
                "corrupted_indexes": corrupted_indexes,
            }
            return JSONResponse(status_code=400, content=error_response)

        # Generate hashes for each image and store them in DB along with image info
        # Pass the results for images and other necessary data to the background task
        background_tasks.add_task(
            process_images_and_store_hashes, results, metadata, job_data
        )

        return JSONResponse(content=finished_response)


# Get the queue length of each API and choose the one with the shortest queue
async def chooseAPI():
    lowest_queue = 9999
    selected_api = None

    for index, api in enumerate(API_IP_List):
        queue_length = global_queue.get(
            index, 9999
        )  # Default to 9999 if API is not in global_queue
        if queue_length < lowest_queue:
            lowest_queue = queue_length
            selected_api = api

    if selected_api is None:
        raise ValueError("No valid API IP found")

    return selected_api


async def enhanced_filter(prompt, pattern, replacement):
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
            prompt = await enhanced_filter(prompt, re.escape(misspelling), correct)

    # # If above is in prompt we grab artist list from DB and remove them if they were in the prompt
    artist_list = []
    try:
        # Connect to the database
        async with await psycopg.AsyncConnection.connect(DSN) as aconn:
            async with aconn.cursor() as acur:
                # Execute the query
                await acur.execute("SELECT Artist FROM excluded_artist")
                rows = await acur.fetchall()

                # Build the artist list
                artist_list = [row[0] for row in rows]

        # Convert the entire prompt to lowercase
        lowercase_prompt = prompt.lower()

        # Check and remove any filtered phrases from the prompt
        for phrase in artist_list:
            # Convert the phrase to lowercase
            lowercase_phrase = phrase.lower()

            # Replace the lowercase phrase in the lowercase prompt
            lowercase_prompt = lowercase_prompt.replace(lowercase_phrase, "")

        # Reconstruct the original case structure of the prompt
        # by iterating over the original prompt and the lowercase prompt
        final_prompt = ""
        for orig_char, lower_char in zip(prompt, lowercase_prompt):
            if lower_char == " " and orig_char != " ":
                # If a character is replaced by space, keep the space
                final_prompt += " "
            else:
                # Otherwise, use the original character
                final_prompt += orig_char

        # Now final_prompt contains the modified prompt with original case structure
        prompt = final_prompt

    except Exception as e:
        print(f"Database error encountered: {e}")

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


async def fortify_default_negative(negative_prompt):
    if "nsfw" in negative_prompt.lower() and "nipples" not in negative_prompt.lower():
        return "nipples, pussy, breasts, " + negative_prompt
    else:
        return negative_prompt


@app.post("/rate_image/")
async def rate_image(job_data: JobData):
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


class DiscordAuthCode(BaseModel):
    code: str


@app.post("/discord_auth/")
async def discord_auth(auth_code: DiscordAuthCode):
    discord_token_url = "https://discord.com/api/oauth2/token"
    client_id = os.environ.get("DISCORD_CLIENT_ID")
    client_secret = os.environ.get("DISCORD_CLIENT_SECRET")
    redirect_uri = os.environ.get("DISCORD_REDIRECT_URI")

    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "authorization_code",
        "code": auth_code.code,
        "redirect_uri": redirect_uri,
    }

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    async with session.post(discord_token_url, data=data, headers=headers) as resp:
        if resp.status != 200:
            raise HTTPException(
                status_code=resp.status, detail="Error in Discord token exchange"
            )
        token_data = await resp.json()

        access_token = token_data.get("access_token")

    discord_guilds_url = "https://discord.com/api/users/@me/guilds"
    headers = {"Authorization": f"Bearer {access_token}"}

    async with session.get(discord_guilds_url, headers=headers) as guild_resp:
        if guild_resp.status != 200:
            raise HTTPException(
                status_code=guild_resp.status,
                detail="Error fetching user guilds from Discord",
            )
        guilds = await guild_resp.json()

    # Fetch the authenticated user's information
    access_token = token_data.get("access_token")
    discord_user_url = "https://discord.com/api/users/@me"
    headers = {"Authorization": f"Bearer {access_token}"}
    async with session.get(discord_user_url, headers=headers) as user_resp:
        if user_resp.status != 200:
            raise HTTPException(
                status_code=user_resp.status,
                detail="Error fetching user data from Discord",
            )
        user_data = await user_resp.json()
        user_id = user_data["id"]  # Get the user's ID

    your_guild_id = "1095514548112461924"  # Replace with your Discord server's ID
    is_member_of_your_guild = any(guild["id"] == your_guild_id for guild in guilds)
    role_ids_to_check = [
        "1097363688995962982",
        "1106031487159128116",
        "1100272052008652922",
    ]

    has_required_role = False  # Replace with your role checking logic
    bot_ip = os.environ.get("DISCORD_BOT_IP")
    # Now make the request to your bot's /check_role endpoint
    async with session.post(
        f"http://{bot_ip}:6965/check_role",
        json={
            "guild_id": your_guild_id,
            "user_id": user_id,
            "role_ids": role_ids_to_check,
        },
        headers={"Authorization": "YourSecretToken"},
    ) as response:
        if response.status != 200:
            raise HTTPException(
                status_code=response.status, detail="Error communicating with the bot"
            )
        data = await response.json()
        has_required_role = data["has_role"]

    return {
        "status": "success",
        "is_member_of_your_guild": is_member_of_your_guild,
        "has_required_role": has_required_role,
    }


# Azure health check, return 200
@app.get("/health_check")
async def health_check():
    return {"status": 200}
