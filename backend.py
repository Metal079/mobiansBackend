import os
import io
import base64
from typing import Optional, Dict
import hashlib
import logging
import asyncio
from datetime import datetime, timedelta
import json
import re
import time
import requests

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
    enable_upscale: Optional[bool] = False


class ImageRequestModel(JobData):
    image: Optional[str] = None
    fast_pass_enabled: Optional[bool] = False


async def refresh_fastpass_cache():
    while True:
        async with await psycopg.AsyncConnection.connect(DSN) as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(
                    "SELECT fastpass_code, expiration_date FROM fastpass_new"
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
        except HTTPException as e:
            # Re-raise the HTTPException to be handled by FastAPI
            raise e
        except Exception as e:
            logging.error(
                "Error occurred while validating FastPassCode (DB might be down)"
            )
            logging.error(str(e))
            raise HTTPException(
                status_code=500,
                detail="An error occurred while validating the FastPassCode. Please try again later.",
            )

    # Filter out prompts
    job_data.prompt, job_data.negative_prompt = await promptFilter(job_data)
    job_data.negative_prompt = await fortify_default_negative(job_data.negative_prompt)

    # Create an instance of ImageRequestModel
    image_request_data = ImageRequestModel(
        **job_data.dict(), fast_pass_enabled=fast_pass_enabled
    )

    async with await psycopg.AsyncConnection.connect(DSN) as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                INSERT INTO generation_queue (
                    id, status, assigned_gpu, prompt, image, image_UUID, mask_image,
                    color_inpaint, control_image, scheduler, steps, negative_prompt,
                    width, height, guidance_scale, seed, batch_size, strength,
                    job_type, model, fast_pass_code, rating, enable_upscale, fast_pass_enabled
                ) VALUES (
                    gen_random_uuid(), 'pending', NULL, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id;
            """,
                (
                    image_request_data.prompt,
                    image_request_data.image,
                    image_request_data.image_UUID,
                    image_request_data.mask_image,
                    image_request_data.color_inpaint,
                    image_request_data.control_image,
                    image_request_data.scheduler,
                    image_request_data.steps,
                    image_request_data.negative_prompt,
                    image_request_data.width,
                    image_request_data.height,
                    image_request_data.guidance_scale,
                    image_request_data.seed,
                    image_request_data.batch_size,
                    image_request_data.strength,
                    image_request_data.job_type,
                    image_request_data.model,
                    image_request_data.fast_pass_code,
                    image_request_data.rating,
                    image_request_data.enable_upscale,
                    fast_pass_enabled,
                ),
            )
            job_id = await acur.fetchone()

    return JSONResponse(content={"job_id": str(job_id[0])})


def decode_base64_to_image(base64_str):
    # Convert base64 string to image
    try:
        image = Image.open(io.BytesIO(base64.b64decode(base64_str.split(",", 1)[1])))
    except:
        image = Image.open(io.BytesIO(base64.b64decode(base64_str.split(",", 1)[0])))

    # Resize image if needed
    # tempAspectRatio = image.width / image.height
    # if tempAspectRatio < 0.8:
    #     image = image.resize((512, 768))
    # elif tempAspectRatio > 1.2:
    #     image = image.resize((768, 512))
    # else:
    #     image = image.resize((512, 512))

    return image


async def increment_fastpass_use_count(fast_pass_code: str):
    # Time the function
    start_time = time.time()
    logging.info("Incrementing FastPassCode use count")

    async with await psycopg.AsyncConnection.connect(DSN) as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE fastpass_new
                SET uses = uses + 1
                WHERE fastpass_code = %s
                """,
                (fast_pass_code,),
            )

    # Time the function
    end_time = time.time()
    logging.info("Time elapsed: " + str(end_time - start_time))


async def set_fastpass_expiration_date(fast_pass_code: str, days_from_today: int):
    # Time the function
    start_time = time.time()
    logging.info("Setting FastPassCode expiration date")

    try:
        # Add days to current date
        expiration_date = datetime.now() + timedelta(days=days_from_today)

        async with await psycopg.AsyncConnection.connect(DSN) as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(
                    """
                    UPDATE fastpass_new
                    SET expiration_date = %s
                    WHERE fastpass_code = %s
                    """,
                    (expiration_date, fast_pass_code),
                )

        # Log the time elapsed
        end_time = time.time()
        logging.info("Time elapsed: {:.2f} seconds".format(end_time - start_time))

    except Exception as e:
        logging.error("Error setting expiration date: %s", e)


async def validate_fastpass(
    fast_pass_code: str, background_tasks: BackgroundTasks
) -> bool:
    # if fast_pass_code in fastpass_cache:
    #     expiration_date = fastpass_cache[fast_pass_code]
    #     if expiration_date is None or expiration_date >= datetime.now():
    #         background_tasks.add_task(increment_fastpass_use_count, fast_pass_code)
    #         return True
    #     else:
    #         return False
    # else:
    # FastPass data not found in cache, query the database
    async with await psycopg.AsyncConnection.connect(DSN) as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT expiration_date, fastpass_days
                FROM fastpass_new
                WHERE fastpass_code = %s
                """,
                (fast_pass_code,),
            )

            row = await acur.fetchone()

            if not row:
                return False

            expiration_date = row[0]
            fastpass_cache[fast_pass_code] = expiration_date  # Store in cache

            # We need to set the code to expire at current date + fastpass_days
            if expiration_date is None:
                days_to_expire = row[1]
                background_tasks.add_task(
                    set_fastpass_expiration_date, fast_pass_code, days_to_expire
                )

            if expiration_date is None or expiration_date >= datetime.now():
                background_tasks.add_task(increment_fastpass_use_count, fast_pass_code)
                return True
            else:
                return False


class GetJobData(BaseModel):
    job_id: str


class JobRetryInfo(BaseModel):
    job_id: str


async def insert_image_hashes(image_hashes, metadata, job_data):
    logging.info("Inserting image hashes")

    insert_query = """
        INSERT INTO hashes (hash, prompt, negative_prompt, seed, cfg, model, created_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    values = [
        (
            image_hashes[i],
            metadata['prompt'],
            metadata['negative_prompt'],
            metadata['seed'],
            metadata['guidance_scale'],
            metadata['model'],
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
        image = decode_base64_to_image(image_results[i])
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
async def get_job(job_data: GetJobData, background_tasks: BackgroundTasks):
    metadata = {}

    async with await psycopg.AsyncConnection.connect(DSN) as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT status, queue_position, finished_images, prompt, negative_prompt, seed, guidance_scale, job_type, model
                FROM vw_generation_queue 
                WHERE id = %s
            """,
                (job_data.job_id,),
            )
            result = await acur.fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Job not found")

    (
        job_status,
        queue_position,
        finished_images,
        metadata["prompt"],
        metadata["negative_prompt"],
        metadata["seed"],
        metadata["guidance_scale"],
        metadata["job_type"],
        metadata["model"],
    ) = result

    if job_status == "completed":

        if finished_images:
            finished_images = finished_images.strip("{}")
            base64_strings = finished_images.split(",")

            # Add watermark and metadata
            watermarked_image_base64 = []
            for i in range(4):
                image = decode_base64_to_image(base64_strings[i])
                watermarked_image_base64.append(
                    await add_image_metadata(image.convert("RGB"), metadata)
                )

            # Generate hashes for each image and store them in DB along with image info
            # Pass the results for images and other necessary data to the background task
            background_tasks.add_task(
                process_images_and_store_hashes,
                watermarked_image_base64,
                metadata,
                job_data,
            )

            return JSONResponse(
                content={"status": "completed", "result": watermarked_image_base64}
            )
        else:
            logging.error(
                f"Job {job_data.job_id} marked as completed but has no finished images"
            )
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Job completed but no images found",
                }
            )
    elif job_status in ["pending", "processing"]:
        return JSONResponse(
            content={
                "status": job_status,
                "queue_position": queue_position,
            }
        )
    else:
        return JSONResponse(
            content={"status": "error", "message": "Unknown job status"}
        )


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
                        watermarked_image_base64 = await add_image_metadata(
                            image.convert("RGB"), metadata
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
