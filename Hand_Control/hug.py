from gradio_client import Client, handle_file
import cv2
import requests
import base64
from openai import OpenAI
from prompt import *
import time

file_path = "1.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

starttime = time.time()

# 将xxxx/test.png替换为你本地图像的绝对路径
base64_image = encode_image(file_path)

client = Client("omlab/VLM-R1-Referral-Expression")
result = client.predict(
		image=  {"url": f"data:image/jpg;base64,{base64_image}"}, #handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
		text=p_content + "请输出图中纸团的抓取代号，不要输出其他任何东西",
		api_name="/predict"
)
print(result)
process = time.time() - starttime
print(process)