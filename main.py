from io import BytesIO
import os

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageColor
import requests


IMAGE_URL = "https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png"
# IMAGE_URL = "https://raw.githubusercontent.com/douglasstarnes/tncommdays/refs/heads/main/maria-lupan-2ojW8P_fB8Y-unsplash.jpg"
# IMAGE_URL = "https://raw.githubusercontent.com/douglasstarnes/tncommdays/refs/heads/main/elena-mozhvilo-UUcnOlfMePU-unsplash.jpg"
COLOR_NAMES = "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta"
COLORS = [
    ImageColor.getrgb(color) for color in COLOR_NAMES
]

load_dotenv()

VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")
VISION_KEY = os.getenv("VISION_KEY")

client = ImageAnalysisClient(
    endpoint=VISION_ENDPOINT,
    credential=AzureKeyCredential(VISION_KEY)
)

result = client.analyze_from_url(
    image_url = IMAGE_URL,
    visual_features=[VisualFeatures.OBJECTS]
)



response = requests.get(IMAGE_URL)

if response.status_code == 200:
    image = Image.open(BytesIO(response.content))

    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default(size=20)

    if result.objects is not None:
        for object in result.objects.list:
            print(object)

        for idx, object in enumerate(result.objects.list):
            coords = object.bounding_box
            draw.rectangle([(coords["x"], coords["y"]), (coords["x"] + coords["w"], coords["y"] + coords["h"])], outline=COLORS[idx], width=3)
            draw.rectangle([(coords["x"], coords["y"] + coords["h"] - 25), (coords["x"] + coords["w"], coords["y"] + coords["h"])], fill=COLORS[idx])
            draw.text((coords["x"] + 2, coords["y"] + coords["h"] - 23), object.tags[0].name, font=font, fill=(0, 0, 0))

    image.save("output_image.png")

