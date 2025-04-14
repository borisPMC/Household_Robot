import cv2
import requests
import base64
from openai import OpenAI
import os
import time
import action

from prompt import p_content

api_key = "sk-73db9647348b4dc2adced1622a4b7271"
url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

file_path = "3.png"

def end():
    cap.release()
    cv2.destroyAllWindows()

def cv2_base64(image):
    base64_str = cv2.imencode('.jpg', image)[1]
    base64_str = base64.b64encode(base64_str)[2:-1]
    return base64_str

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
#process_time = 0
#answer = ""
"运行qvq模型，返回抓取类型和运行时间"
def qvq(img):
    starttime = time.time()
    #  base 64 编码格式
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # 将xxxx/test.png替换为你本地图像的绝对路径
    base64_image = encode_image("tmp.jpg")
    #base64_image = cv2_base64(img)

    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key= api_key,
        base_url=url,
    )

    obj_class = ""
    obj_len = 30
    obj_wid = 15


    completion = client.chat.completions.create(
        #model="qvq-72b-preview",
        model="qwen2.5-vl-7b-instruct",
        messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": p_content}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                        # PNG图像：  f"data:image/png;base64,{base64_image}"
                        # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                        # WEBP图像： f"data:image/webp;base64,{base64_image}"
                        "image_url": {"url": f"data:image/jpg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": "请输出图正中央物体适合的抓取代号"}# + obj_class + "长" +str(obj_len) + "厘米宽"+ str(obj_wid) +"厘米，请输出它的抓取代号"},
                ],
            }
        ],
    )
    answer = completion.choices[0].message.content
    print(answer)
    process_time = time.time() - starttime
    print("process time:" + str(process_time))
    return answer,'{:.2f}'.format(process_time)


if __name__ == "__main__":
    print("访问相机")
    cap = cv2.VideoCapture(0)
    print("访问成功")
    showtext = "press n to predict"
    while(1):
        # get a frame
        ret, frame = cap.read()
        img = frame.copy()



        # show a frame
        cv2.putText(frame, showtext, (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 200), 2)

        cv2.imshow("capture", frame)

        keyvalue = cv2.waitKey(1)

        if keyvalue == 27:
            break
        elif keyvalue == ord('n'):
            cv2.imwrite("tmp.jpg", img)
            answer, process_time = qvq(img)
            if answer != "":
                print("changed")
                action.take_action(int(answer),0,0,0)
                showtext = "gesture:" + str(answer) + " time:" + str(process_time)
                cv2.putText(frame, showtext, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 200), 2)
    end()


