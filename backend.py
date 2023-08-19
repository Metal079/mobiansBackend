import os
import io
import base64
import requests
import time
from typing import Optional, List
import time
import hashlib
import logging
import random
import asyncio
from datetime import datetime
from io import BytesIO
import uuid
import json

import aiohttp
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from pydantic import BaseModel
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from dotenv import load_dotenv
import redis
from redis.asyncio import Redis
from redis.backoff import ExponentialBackoff
from redis.retry import Retry
from redis.exceptions import (
   BusyLoadingError,
   ConnectionError,
   TimeoutError
)
import aioodbc
from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from pywebpush import webpush, WebPushException

logging.basicConfig(level=logging.INFO)  # Configure logging

# Run 3 retries with exponential backoff strategy
retry = Retry(ExponentialBackoff(), 3)
load_dotenv()
REDISHOST = os.environ.get('REDISHOST')

DBHOST = os.environ.get('DBHOST')
DBNAME = os.environ.get('DBNAME')
DBUSER = os.environ.get('DBUSER')
DBPASS = os.environ.get('DBPASS')
driver= '{ODBC Driver 17 for SQL Server}'
# cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
dsn = f'DRIVER={driver};SERVER={DBHOST};DATABASE={DBNAME};UID={DBUSER};PWD={DBPASS}'

API_IP_List = os.environ.get('API_IP_List').split(' ')

VAPID_PUBLIC_KEY = os.environ.get('VAPID_PUBLIC_KEY')
VAPID_PRIVATE_KEY = os.environ.get('VAPID_PRIVATE_KEY')
VAPID_CLAIMS = os.environ.get('VAPID_CLAIMS')
subscriptions = []  # Store your subscription objects here

app = FastAPI()

session = None

@app.on_event("startup")
async def startup_event():
    global session
    global blob_service_client
    global r

    session = aiohttp.ClientSession()
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    r = Redis(host=REDISHOST, port=6379, db=0, retry=retry, retry_on_error=[BusyLoadingError, ConnectionError, TimeoutError]) #, decode_responses=True

@app.on_event("shutdown")
async def shutdown_event():
    await session.close()

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

@app.post("/submit_job/")
async def submit_job(job_data: JobData):
    # Check if FastPassCode is valid and non-expired
    fast_pass_enabled = False
    if job_data.fast_pass_code:
        is_valid = await validate_fastpass(job_data.fast_pass_code)
        if is_valid:
            fast_pass_enabled = True
        else:
            raise HTTPException(status_code=400, detail="Invalid or expired FastPassCode. Please fix/remove the FastPassCode and try again.")

    # Filter out prompts 
    job_data.prompt, job_data.negative_prompt = promptFilter(job_data)
    job_data.negative_prompt = fortify_default_negative(job_data.negative_prompt)

    API_IP = await chooseAPI('txt2img')

    # Do img2img filtering if it's an img2img request
    if job_data.job_type == 'img2img' or job_data.job_type == 'inpainting':
        # Convert base64 string to image to remove alpha channel if needed
        try:
            received_image = Image.open(io.BytesIO(base64.b64decode(job_data.image.split(",", 1)[0])))
        except:
            received_image = Image.open(io.BytesIO(base64.b64decode(job_data.image.split(",", 1)[1])))
        if received_image.mode == 'RGBA':
            buffer = io.BytesIO()
            
            # Separate alpha channel and add white background
            background = Image.new('RGBA', received_image.size, (255, 255, 255))
            alpha_composite = Image.alpha_composite(background, received_image).convert('RGB')
            alpha_composite.save(buffer, format='PNG')

            # Convert received_image back to base64 string
            buffer.seek(0)
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            job_data.image = encoded_image

        # Resize image if needed
        # NOTE THE IMAGE PROPERTY IS A BASE64 STRING EVEN THOUGH IT PROBABLY SHOULD BE AN IMAGE OBJECT
        try:
            init_image = Image.open(io.BytesIO(base64.b64decode(job_data.image.split(",", 1)[0])))
        except:
            init_image = Image.open(io.BytesIO(base64.b64decode(job_data.image.split(",", 1)[1])))
        tempAspectRatio = init_image.width / init_image.height
        if tempAspectRatio < 0.8:
            init_image = init_image.resize((512, 768))
        elif tempAspectRatio > 1.2:
            init_image = init_image.resize((768, 512))
        else:
            init_image = init_image.resize((512, 512))

        # Save resized image to a BytesIO object
        buffer = io.BytesIO()
        init_image.save(buffer, format='PNG')
        buffer.seek(0)

        # Encode BytesIO object to base64
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        job_data.image = encoded_image

    # Do the same for mask image
    if job_data.mask_image:
        try:
            received_image = Image.open(io.BytesIO(base64.b64decode(job_data.mask_image.split(",", 1)[0])))
        except:
            received_image = Image.open(io.BytesIO(base64.b64decode(job_data.mask_image.split(",", 1)[1])))
        if received_image.mode == 'RGBA':
            buffer = io.BytesIO()
            
            # Separate alpha channel and add white background
            background = Image.new('RGBA', received_image.size, (255, 255, 255))
            alpha_composite = Image.alpha_composite(background, received_image).convert('RGB')
            alpha_composite.save(buffer, format='PNG')

            # Convert received_image back to base64 string
            buffer.seek(0)
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            job_data.mask_image = encoded_image

        # Resize image if needed
        try:
            init_image = Image.open(io.BytesIO(base64.b64decode(job_data.mask_image.split(",", 1)[0])))
        except:
            init_image = Image.open(io.BytesIO(base64.b64decode(job_data.mask_image.split(",", 1)[1])))
        tempAspectRatio = init_image.width / init_image.height
        if tempAspectRatio < 0.8:
            init_image = init_image.resize((512, 768))
        elif tempAspectRatio > 1.2:
            init_image = init_image.resize((768, 512))
        else:
            init_image = init_image.resize((512, 512))

        # Save resized image to a BytesIO object
        buffer = io.BytesIO()
        init_image.save(buffer, format='PNG')
        buffer.seek(0)

        # Encode BytesIO object to base64
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        job_data.mask_image = encoded_image


    # Create an instance of ImageRequestModel
    image_request_data = ImageRequestModel(**job_data.dict(), fast_pass_enabled=fast_pass_enabled)

    # Try using the requested API, if it fails, use the other one
    try:
        response = requests.post(url=f'{API_IP}/submit_job/', json=image_request_data.dict())
    except:
        API_IP = chooseAPI('txt2img', [API_IP])
        response = requests.post(url=f'{API_IP}/submit_job/', json=image_request_data.dict())

    attempts = 0
    while response.status_code != 200 and attempts < 3:
        API_IP = chooseAPI('txt2img', [API_IP])
        print(f"got error: {response.status_code} for submit_job, api: {API_IP}")
        attempts += 1
        response = requests.post(url=f'{API_IP}/submit_job/', json=image_request_data.dict())
        time.sleep(1)

    returned_data = response.json()
    # Get index of API_IP in API_IP_List
    for i in range(len(API_IP_List)):
        if API_IP_List[i] == API_IP:
            returned_data['API_IP'] = i

    return JSONResponse(content=returned_data)

async def validate_fastpass(fast_pass_code: str) -> bool:
    async with aioodbc.connect(dsn=dsn) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT ExpirationDate
                FROM FastPass
                WHERE FastPassCode = ?
            """, fast_pass_code)

            row = await cursor.fetchone()
            if not row:
                return False

            expiration_date = row[0]
            if expiration_date is None:
                return True
            elif expiration_date < datetime.now():
                return False

    return True

class GetJobData(BaseModel):
    job_id: str
    API_IP: int

class JobRetryInfo(BaseModel):
    job_id: str
    indexes: List[int]

async def call_get_job(job_id: str, API_IP: str):
    async with session.get(url=f"{API_IP}/get_job/{job_id}") as response:
        return await response.json()

@app.post("/get_job/")
async def get_job(job_data: GetJobData):
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
                logging.error(f"Max retries exceeded when making GET request, JOB: {job_data.job_id}")
                return JSONResponse(content={'message': 'Error occurred while making GET request'}, status_code=500)
            else:
                # Calculate next sleep time
                sleep_time = MIN_DELAY * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, MAX_DELAY)

                logging.error(e)
                logging.error(f"Exception occurred when making GET request, JOB: {job_data.job_id}. Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)

    try:
        if response['status'] == 'completed':
            async with r.pipeline() as pipe:
                # Fetch images from Redis
                finished_response = {'status': 'completed', 'result': []}
                metadata_json = await r.get(f"job:{job_data.job_id}:metadata") # Changed to async
                metadata = JobData.parse_raw(metadata_json)

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
                    image_bytes = results[2*i]
                    fetched_checksum = results[2*i + 1]
                    
                    if image_bytes is not None and fetched_checksum is not None:
                        computed_checksum = hashlib.sha256(image_bytes).hexdigest()
                        if fetched_checksum.decode() != computed_checksum:
                            corrupted_indexes.append(i)

                # If there are corrupted images, resend them
                if corrupted_indexes:
                    logging.info(f"Corrupted images detected for job {job_data.job_id}, resending corrupted images")
                    retry_info = JobRetryInfo(job_id=job_data.job_id, indexes=corrupted_indexes)
                    requests.get(url=f"{API_IP_List[job_data.API_IP]}/resend_images/{job_data.job_id}", json=retry_info.dict())

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
                        image_bytes = results[2*i]
                        fetched_checksum = results[2*i + 1]
                        
                        if image_bytes is not None and fetched_checksum is not None:
                            computed_checksum = hashlib.sha256(image_bytes).hexdigest()
                            if fetched_checksum.decode() == computed_checksum:
                                image = Image.open(io.BytesIO(image_bytes))

                                # Add watermark and metadata
                                watermarked_image = add_watermark(image.convert("RGB"))
                                watermarked_image_base64 = add_image_metadata(watermarked_image, metadata)
                                finished_response['result'].append(watermarked_image_base64)
                        else:
                            logging.error(f"Corrupted image STILL detected for job {job_data.job_id}")
                    
                    if len(finished_response['result']) == 4:
                        break
                    else:
                        attempts += 1

                return JSONResponse(content=finished_response)


    except Exception as e:
        # logging.error(f"got error: {response.status_code} for retrieve_job on job {job_data.job_id}, api: {API_IP_List[job_data.API_IP]}")
        # logging.error(f"response: {response.text}")
        logging.error(f"response: {response}")
        logging.error(f"Exception: {e}")
        logging.error(f"Exception happened on get_job")

    return JSONResponse(content=response)

async def call_api(api, session):
    try:
        async with session.get(url=f"{api}/get_queue_length/", timeout=5) as response:
            return await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"API {api} is down. Error: {str(e)}")
        return {'queue_length': 9999, 'api': api}

# Get the queue length of each API and choose the one with the shortest queue
async def chooseAPI(generateType, triedAPIs=[]):
    API_queue_length_list = []
    current_lowest_queue = 9999

    global session
    tasks = []
    for api in API_IP_List:
        if api not in triedAPIs:
            task = asyncio.create_task(call_api(api, session))
            tasks.append((api, task))

    results = await asyncio.gather(*[task for api, task in tasks], return_exceptions=True)

    for api, result in zip([api for api, task in tasks], results):
        if isinstance(result, Exception):
            print(f"API {api} is down. Error: {str(result)}")
            continue

        queue_length = result['queue_length']
        API_queue_length_list.append(queue_length)
        print(f"API {api} queue length: {queue_length}")

        if queue_length < current_lowest_queue:
            current_lowest_queue = queue_length
            lowest_index = API_IP_List.index(api)
    
    return API_IP_List[lowest_index]

def promptFilter(data):
    prompt = data.prompt
    negative_prompt = data.negative_prompt

    character_list = ['cream the rabbit', 
                      'rosy the rascal',
                      'sage',
                      'maria robotnik',
                      'marine the raccoon',
                      'charmy the bee'
                      ]
    
    censored_tags = ['breast',
                     'nipples',
                     'pussy',
                     'nsfw',
                     'nudity',
                     'naked',
                     'loli',
                     'nude',
                     'ass',
                     'rape',
                     'sex',
                     'boob',
                     'sex',
                     'busty',
                     'tits',
                     'thigh',
                     'thick',
                     'underwear',
                     'panties',
                     'upskirt',
                     'cum',
                    'dick',
                    'topless',
                    'penis',
                    'blowjob',
                    'ahegao',
                    'nude',
                    'hips',
                    'areola',
                    'pantyhose',
                    'creampie',
                    'position',
                    'wet',
                    'autocunnilingus',
                    'squirting',
                    'straddling',
                    'girl on top',
                    'reverse cowgirl',
                    'feet',
                    'toes',
                    'footjob',
                    'vagina',
                    'clitoris',
                    'furry with non-furry',
                    'spread legs',
                    'navel',
                    'bimbo',
                    'fishnet',
                    'hourglass figure',
                    'slut',
                    'interspecies',
                    'hetero',
                    'tongue',
                    'saliva'
                    'anal',
                    'penetration',
                    'anus',
                    'erection',
                    'masterbation',
                    'butt',
                    'thighhighs',
                    'lube',
                    'lingerie',
                    'bent over',
                    'doggystyle',
                    'sexy',
                    'Areolae',
                    'exhibitionism',
                    'bottomless',
                    'shirt lift',
                    'no bra',
                    'curvy',
                    'groin',
                    'clothes lift',
                    'stomach',
                    'spreading legs',
                    'hentai',
                    'penetrated',
                    'masturbating',
                    'masturbate',
                    'horny',
                    'orgasm',
                    'fingering',
                    'voluptuous',
                    'sperm',
                    'handjob',
                    'gangbang',
                    'ejaculation',
                    'uncensored',
                    'Lifting Skirt',
                    'mooning',
                    'hindquarters',
                    'presenting',
                    'porn',
                    'latex',
                    'fellatio',
                    'oral',
                    'open legs',
                    'spread wide',
                    'fucked',
                    'fucking',
                    'g-string',
                    'seductive gaze',
                    'dress lift',
                    'cleavage',
                    'provocative',
                    'venus body',
                    'revealing clothes',
                    'oppai',
                    'milf',
                    'wardrobe malfunction',
                    'clothing aside',
                    'micro bikini',
                    'thong',
                    'gstring',
                    'mating'
                     ]

    # If character is in prompt, filter out censored tags from prompt
    if any(character in prompt.lower() for character in character_list):
        for tag in censored_tags:
            prompt = prompt.lower().replace(tag.lower(), '')
        negative_prompt = "nipples, sexy, breasts, nude, " + negative_prompt
        logging.error(prompt)
            
    return prompt, negative_prompt

def fortify_default_negative(negative_prompt):
    if "nsfw" in negative_prompt.lower() and "nipples" not in negative_prompt.lower():
        return "nipples, pussy, breasts, " + negative_prompt
    else:
        return negative_prompt
    

def filter_seed(data):
    seed = data['seed']
    try:
        if not isinstance(data['data']['seed'], int):
            seed = 999999
        elif data['data']['seed'] > 999999 | data['data']['seed'] < 1:
            seed = 999999
    except:
        seed = 1

    return seed

def add_watermark(image):
    # Create watermark image
    watermark_text = "Mobians.ai"
    opacity = 128
    watermark = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(watermark)

    # Provide the correct path to the font file
    font_file_path = r'fonts/Roboto-Medium.ttf'  
    font = ImageFont.truetype(font_file_path, 25)
    draw.text((10, 10), watermark_text, font=font, fill=(255, 255, 255, opacity))

    # Overlay watermark on the original image
    image_with_watermark = Image.alpha_composite(image.convert("RGBA"), watermark)
    return image_with_watermark

def add_image_metadata(image, request_data):
    img_io = io.BytesIO()

    image_with_watermark = add_watermark(image)

    # Add metadata
    metadata = PngImagePlugin.PngInfo()
    try:
        # Add disclamer to metadata if job is not txt2img
        if request_data.job_type != "txt2img":
            metadata.add_text("NOTE", "The image was not generated purely using txt2img, using the info below may not give you the same results.")

        metadata.add_text("prompt", request_data.prompt)
        request_data.negative_prompt = request_data.negative_prompt.replace("admin", "")
        metadata.add_text("negative_prompt", request_data.negative_prompt)
        metadata.add_text("seed", str(request_data.seed))
        metadata.add_text("cfg", str(request_data.guidance_scale))
        metadata.add_text("job_type", request_data.job_type)    
    except:
        # log to text file
        logging.error(f"Error adding metadata to image")
        with open("error_log.txt", "a") as f:
            f.write(f"Error adding metadata to image: {request_data}\n")
    metadata.add_text("model", "Mobians.ai / SonicDiffusionV3Beta4")
    metadata.add_text("Disclaimer", "The image is generated by Mobians.ai. The image is not real and is generated by an AI.")

    image_with_watermark.save(img_io, format='PNG', pnginfo=metadata)
    #image_with_watermark.save(img_io, format='WEBP', quality=95)
    img_io.seek(0)
    base64_image = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return base64_image

async def upload_blob(blob_service_client, container_name, blob_name, data):
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)

    # Check if the blob exists
    if await blob_client.exists():
        try:
            # Delete the existing blob
            await blob_client.delete_blob()
        except Exception as e:
            print(f"An error occurred while deleting blob {blob_name}: {e}")
    else:
        try:
            # Upload the new blob
            await blob_client.upload_blob(data)
            return blob_client.url
        except Exception as e:
            print(f"An error occurred while uploading blob {blob_name}: {e}")

async def delete_and_insert_image_metadata(image_details, blob_url, dsn, rating, uuid):
    async with aioodbc.connect(dsn=dsn) as conn:
        async with conn.cursor() as cursor:
            # Check if the UUID already exists
            check_query = "SELECT * FROM UserRatings WHERE FileName = ?"
            await cursor.execute(check_query, (uuid,))
            existing_record = await cursor.fetchone()

            if existing_record:
                # If the UUID exists, delete the existing record
                delete_query = "DELETE FROM UserRatings WHERE FileName = ?"
                await cursor.execute(delete_query, (uuid,))
                return "deleted"

            else:
                # Insert a new record
                insert_query = """
                    INSERT INTO UserRatings (Prompt, NegativePrompt, Seed, CFG, FileName, RateDate, UserRating, JobType, AzureBlobURL)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                file_name = uuid
                rate_date = datetime.now()
                await cursor.execute(insert_query, (image_details['prompt'], image_details['negative_prompt'], image_details['seed'], image_details['cfg'], file_name, rate_date, rating, image_details['job_type'], blob_url))

                await conn.commit()
                return "inserted"

@app.post("/rate_image/")
async def rate_image(job_data: JobData):
    # Decode base64 image and convert it to bytes
    try:
        image_bytes = base64.b64decode(job_data.image.split(",", 1)[0])
    except:
        image_bytes = base64.b64decode(job_data.image.split(",", 1)[1])

    image = Image.open(BytesIO(image_bytes))

    info = image.info

    # Get image metadata
    image_details = {}
    for key, value in info.items():
        image_details[key] = value

    # Generate a random filename
    image_name = job_data.image_UUID

    # Upload to Azure Blob Storage
    container_name = "mobiansratings"
    blob_url = await upload_blob(blob_service_client, container_name, image_name, image_bytes)

    # Insert image metadata into database
    db_status = await delete_and_insert_image_metadata(image_details, blob_url, dsn, rating=job_data.rating, uuid=image_name)

    return JSONResponse({"message": f"{db_status}"})

class Subscription(BaseModel):
    endpoint: str
    expirationTime: Optional[str]
    keys: dict

@app.post("/subscribe")
async def subscribe(subscription: Subscription):
    subscriptions.append(subscription.dict())
    return {"status": "subscribed"}

@app.get("/send_notification")
async def send_notification():
    for subscription in subscriptions:
        try:
            webpush(
                subscription_info=subscription,
                data=json.dumps({"message": "Your image is ready!"}),
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={
                    "sub": "mailto:your_email@example.com"
                }
            )
        except WebPushException as e:
            print("Failed to send notification:", repr(e))
            return {"status": "failed", "detail": repr(e)}

    return {"status": "sent"}

