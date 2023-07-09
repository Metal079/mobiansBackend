import os
import json
import io
import base64
import requests
import random
import time
from typing import Optional

from fastapi import FastAPI, Request, Depends, status, Response, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from pydantic import BaseModel
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from dotenv import load_dotenv
load_dotenv()


API_IP_List = os.environ.get('API_IP_List').split(' ')

app = FastAPI()


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

    API_IP = chooseAPI('txt2img')

    # Do img2img filtering if it's an img2img request
    if job_data.job_type == 'img2img':
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

@app.post("/get_job/")
async def get_job(job_data: GetJobData):
    response = requests.get(url=f"{API_IP_List[job_data.API_IP]}/get_job/{job_data.job_id}", json=job_data.dict())

    if response.status_code != 200:
        print(f"got error: {response.status_code} for retrieve_job on job {job_data.job_id}, api: {API_IP_List[job_data.API_IP]}")
        print(f"response: {response.text}")
        response = requests.get(url=f"{API_IP_List[job_data.API_IP]}/get_job/{job_data.job_id}", json=job_data.dict())
        time.sleep(1)

        # Try it one more time
        response = requests.get(url=f"{API_IP_List[job_data.API_IP]}/get_job/{job_data.job_id}", json=job_data.dict())

    # Add watermark and metadata to images if they're ready
    metadata = "placeholder"
    try:
        if response.json()['status'] == 'completed':
            # Add watermark to images
            watermarked_images = []
            finished_response = {'status': 'completed', 'result': []}

            for index, image in enumerate(response.json()['result']):
                watermarked_images.append(add_watermark(Image.open(io.BytesIO(base64.b64decode(image)))).convert("RGB"))
                finished_response['result'].append(add_image_metadata(watermarked_images[index], metadata))

            return JSONResponse(content=finished_response, status_code=response.status_code)
    #print the error message
    except:
        print(f"got error: {response.status_code} for retrieve_job on job {job_data.job_id}, api: {API_IP_List[job_data.API_IP]}")
        print(f"response: {response.text}")
        print(f"response.json(): {response.json()}")
        return JSONResponse(content=response.json(), status_code=response.status_code)

    return JSONResponse(content=response.json(), status_code=response.status_code)

# Get the queue length of each API and choose the one with the shortest queue
def chooseAPI(generateType, triedAPIs=[]):
    API_queue_length_list = []
    current_lowest_queue = 9999
    for index, api in enumerate(API_IP_List):
        try:
            if api not in triedAPIs:
                response = requests.get(url=f'{api}/get_queue_length/')
                API_queue_length_list.append(response.json()['queue_length'])
                print(f"API {api} queue length: {response.json()['queue_length']}")

                if response.json()['queue_length'] < current_lowest_queue:
                    current_lowest_queue = response.json()['queue_length']
                    lowest_index = index
        except:
            print(f"API {api} is down")
            continue
    
    return API_IP_List[lowest_index]

def promptFilter(data):
    prompt = data.prompt
    negative_prompt = data.negative_prompt

    character_list = ['cream the rabbit', 
                      'rosy the rascal',
                      'sage',
                      'maria robotnik',
                      'marine the raccoon',
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
                     'sexy',
                     'busty',
                     'tits',
                     'thighs',
                     'thick',
                     'underwear',
                     'panties',
                     'upskirt',
                     'cum'
                     ]

    # If character is in prompt, filter out censored tags from prompt
    if any(character in prompt.lower() for character in character_list):
        for tag in censored_tags:
            prompt = prompt.replace(tag, '')
        negative_prompt = "nipples, pussy, breasts, " + negative_prompt
            
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
    # Add disclamer to metadata if job is not txt2img
    # if job_type != "txt2img":
    #     metadata.add_text("NOTE", "The image was not generated purely using txt2img, using the info below may not give you the same results.")
    # metadata.add_text("prompt", request_data.data['prompt'])
    # metadata.add_text("negative_prompt", request_data.data['negative_prompt'])
    # metadata.add_text("seed", str(request_data.data['seed']))
    # metadata.add_text("cfg", str(request_data.data['guidance_scale']))
    metadata.add_text("Disclaimer", "The image is generated by Mobians.ai. The image is not real and is generated by an AI.")

    image_with_watermark.save(img_io, format='PNG', pnginfo=metadata)
    #image_with_watermark.save(img_io, format='WEBP', quality=95)
    img_io.seek(0)
    base64_image = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return base64_image