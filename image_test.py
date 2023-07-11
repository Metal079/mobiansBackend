
import redis
from PIL import Image, ImageDraw, ImageFont
import io

r = redis.Redis(host='7.tcp.ngrok.io', port=21658, db=0)


key = f"job:{'a87bf9ba-5d8f-474d-a3cc-5f3a43e51912'}:image:{0}"
image1 = r.get(key)
image1 = Image.open(io.BytesIO(image1))
image1.show()

key = f"job:{'a87bf9ba-5d8f-474d-a3cc-5f3a43e51912'}:image:{4}"
image3 = r.get(key)
image3 = Image.open(io.BytesIO(image3))
image3.show()
