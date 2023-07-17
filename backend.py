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

import aiohttp
from fastapi import FastAPI
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
from redis.backoff import ExponentialBackoff
from redis.retry import Retry
from redis.exceptions import (
   BusyLoadingError,
   ConnectionError,
   TimeoutError
)

logging.basicConfig(level=logging.INFO)  # Configure logging

# r = redis.Redis(host='7.tcp.ngrok.io', port=21658, db=0)
# Run 3 retries with exponential backoff strategy
retry = Retry(ExponentialBackoff(), 3)
r = redis.Redis(host='76.157.184.213', port=6379, db=0, retry=retry, retry_on_error=[BusyLoadingError, ConnectionError, TimeoutError])
load_dotenv()


API_IP_List = os.environ.get('API_IP_List').split(' ')

app = FastAPI()

session = None

@app.on_event("startup")
async def startup_event():
    global session
    session = aiohttp.ClientSession()

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

class JobData(BaseModel):
    prompt: str
    image: Optional[str] = None
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

@app.post("/submit_job/")
async def submit_job(job_data: JobData):
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

    # Try using the requested API, if it fails, use the other one
    try:
        response = requests.post(url=f'{API_IP}/submit_job/', json=job_data.dict())
    except:
        API_IP = chooseAPI('txt2img', [API_IP])
        response = requests.post(url=f'{API_IP}/submit_job/', json=job_data.dict())

    attempts = 0
    while response.status_code != 200 and attempts < 3:
        API_IP = chooseAPI('txt2img', [API_IP])
        print(f"got error: {response.status_code} for submit_job, api: {API_IP}")
        attempts += 1
        response = requests.post(url=f'{API_IP}/submit_job/', json=job_data.dict())
        time.sleep(1)

    returned_data = response.json()
    # Get index of API_IP in API_IP_List
    for i in range(len(API_IP_List)):
        if API_IP_List[i] == API_IP:
            returned_data['API_IP'] = i

    return JSONResponse(content=returned_data)

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
            # Fetch images from Redis
            finished_response = {'status': 'completed', 'result': []}
            metadata_json = r.get(f"job:{job_data.job_id}:metadata")
            metadata = JobData.parse_raw(metadata_json)

            # First pass: Identify corrupted images
            corrupted_indexes = []
            for i in range(4):
                key = f"job:{job_data.job_id}:image:{i}"
                image_bytes = r.get(key)
                fetched_checksum = r.get(f"job:{job_data.job_id}:image:{i}:checksum")
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
            for i in range(4):
                key = f"job:{job_data.job_id}:image:{i}"
                attempts = 0
                while attempts < 2:
                    image_bytes = r.get(key)
                    fetched_checksum = r.get(f"job:{job_data.job_id}:image:{i}:checksum")
                    if image_bytes is not None and fetched_checksum is not None:
                        computed_checksum = hashlib.sha256(image_bytes).hexdigest()
                        if fetched_checksum.decode() == computed_checksum:
                            image = Image.open(io.BytesIO(image_bytes))

                            # Add watermark and metadata
                            watermarked_image = add_watermark(image.convert("RGB"))
                            watermarked_image_base64 = add_image_metadata(watermarked_image, metadata)
                            finished_response['result'].append(watermarked_image_base64)
                            break  # valid image, no need for further attempts
                    else:
                        logging.error(f"Corrupted image STILL detected for job {job_data.job_id}")
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
    async with session.get(url=f"{api}/get_queue_length/") as response:
        return await response.text()

# Get the queue length of each API and choose the one with the shortest queue
async def chooseAPI(generateType, triedAPIs=[]):
    API_queue_length_list = []
    current_lowest_queue = 9999

    # global session
    # tasks = []
    for index, api in enumerate(API_IP_List):
        try:
            if api not in triedAPIs:
                # task = asyncio.create_task(call_api(api, session))
                # tasks.append(task)
                response = requests.get(url=f'{api}/get_queue_length/', timeout=10)
                API_queue_length_list.append(response.json()['queue_length'])
                print(f"API {api} queue length: {response.json()['queue_length']}")

                if response.json()['queue_length'] < current_lowest_queue:
                    current_lowest_queue = response.json()['queue_length']
                    lowest_index = index
        except:
            print(f"API {api} is down")
            continue

    # thing = await asyncio.gather(*tasks)
    
    return API_IP_List[lowest_index]

# def chooseAPI(generateType, triedAPIs=[]):
#     API_queue_length_list = []
#     current_lowest_queue = 9999
#     for index, api in enumerate(API_IP_List):
#         try:
#             if api not in triedAPIs:
#                 response = requests.get(url=f'{api}/get_queue_length/')
#                 API_queue_length_list.append(response.json()['queue_length'])
#                 print(f"API {api} queue length: {response.json()['queue_length']}")

#                 if response.json()['queue_length'] < current_lowest_queue:
#                     current_lowest_queue = response.json()['queue_length']
#                     lowest_index = index
#         except:
#             print(f"API {api} is down")
#             continue
    
#     return API_IP_List[lowest_index]

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
                     'thighs',
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
                    'uncensored'
                     ]

    # If character is in prompt, filter out censored tags from prompt
    if any(character in prompt.lower() for character in character_list):
        for tag in censored_tags:
            prompt = prompt.replace(tag, '')
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
        # if job_type != "txt2img":
        #     metadata.add_text("NOTE", "The image was not generated purely using txt2img, using the info below may not give you the same results.")
        metadata.add_text("prompt", request_data.prompt)
        request_data.negative_prompt = request_data.negative_prompt.replace("admin", "")
        metadata.add_text("negative_prompt", request_data.negative_prompt)
        metadata.add_text("seed", str(request_data.seed))
        metadata.add_text("cfg", str(request_data.guidance_scale))
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