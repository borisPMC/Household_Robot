from typing import Optional, Union
import cv2
import torch
from base64 import b64decode
import os
import shutil
import time
from PIL import Image
# from IPython.display import display, Javascript, Image
from js2py import eval_js
from ultralytics import YOLO
from paddleocr import PaddleOCR

"""
META DATA SECTION
"""
YAML_CONFIG = """
path: ./Object_Detection  # Root of Dataset
train: images/train  # Path to training set from path
val: images/val      # Path to validation set from path
nc: 1                # Number of classes
names:
  - medicine_bottle  # Class name
iou_type: siou     # Optional parameters (Decomment if needed)
"""

YAML_FPATH = "./Object_Detection/content/info.yaml"
LIVE_PHOTO_FPATH = "./temp"
MAX_ATTEMPTS = 3

# def get_loss_by_iou(coord: list[8]):

#     b2_x1, b2_x2, b1_x1, b1_x2 = coord[0], coord[1], coord[2], coord[3]
#     b2_y1, b2_y2, b1_y1, b1_y2 = coord[4], coord[5], coord[6], coord[7]

#     # Initialize iou with an initial value
#     s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5  # 真实框和预测框中心点的宽度差
#     s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5  # 真实框和预测框中心点的高度差
#     sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)  # 真实框和预测框中心点的距离
#     sin_alpha_1 = torch.abs(s_cw) / sigma  # 真实框和预测框中心点的夹角β
#     sin_alpha_2 = torch.abs(s_ch) / sigma  # 真实框和预测框中心点的夹角α
#     threshold = pow(2, 0.5) / 2  # 夹角阈值
#     sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)  # α大于45°则考虑优化β，否则优化α
#     angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)  # 角度损失
#     rho_x = (s_cw / cw) ** 2
#     rho_y = (s_ch / ch) ** 2
#     gamma = angle_cost - 2
#     distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)  # 距离损失
#     omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
#     omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
#     shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)  # 形状损失
#     iou = iou - 0.5 * (distance_cost + shape_cost)  # siou

#     loss = 1.0 - iou

def iou_type(iou_type: str, coord: list[8]) -> Union[str, float]:
    
    result = None
    match iou_type:
        case "iou":
            print("Using IOU")
            result = iou_type
        case "siou":
            print("Using SIOU")
            result = iou_type
        case _:
            print("Unknown IOU type. Calculate Manually")
            # result =get_loss_by_iou(coord)
    
    return result

def write_yaml_config(fpath: str, content: str) -> None:
    try:
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w", encoding="utf-8") as file:
            file.write(content)
    except FileNotFoundError:
        print("YAML File not found!")
    finally:
        return

# Define auto-capture function with 5-second delay
def auto_capture_photo(fpath='./temp/photo.jpg', quality=0.8):
    js = '''
        async function autoCapture(quality) {
            // Create UI elements
            const div = document.createElement('div');
            const countdown = document.createElement('div');
            countdown.style.fontSize = '24px';
            countdown.style.fontWeight = 'bold';
            countdown.style.color = 'red';
            div.appendChild(countdown);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Resize the output to fit the video element
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            // Show countdown
            let seconds = 5;
            countdown.textContent = `Auto-capturing in ${seconds}...`;

            // Countdown timer
            await new Promise((resolve) => {
                const timer = setInterval(() => {
                    seconds--;
                    countdown.textContent = `Auto-capturing in ${seconds}...`;
                    if (seconds <= 0) {
                        clearInterval(timer);
                        resolve();
                    }
                }, 1000);
            });

            // Capture photo
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            div.remove();
            return canvas.toDataURL('image/jpeg', quality);
        }
    '''
    # display(js)
    data = eval_js('autoCapture({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(fpath, 'wb') as f:
        f.write(binary)
    return fpath

def detect_medicine_exist(im_model) -> Optional[list]:

    # Custom Exception for Expected Error (No bottle case)
    class BottleNotFoundException(Exception):
        pass

    # Initialise param
    conf = []
    coord = [dict] # Format for each item: {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

    try:
        # Auto-capture photo with 5-second countdown
        fname = auto_capture_photo()

        if not fname:
            raise ValueError("Failed to generate photo file")

        print(f'Photo automatically captured and saved to: {fname}')

        # Run inference
        results = im_model(fname)

        # Process results
        if not results or not results[0].boxes:
            raise BottleNotFoundException()

        # Successful detection
        print("\n✅ Medicine bottle detected! Details:")
        for box in results[0].boxes: # Multiple bottles are possible
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            coord.append({"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max})
            conf.append(box.conf.item())
            print(f"Bounding box: ({x_min}, {y_min}, {x_max}, {y_max})")
            print(f"Confidence: {conf*100:.1f}%")

            image = cv2.imread(fname)
    
    except BottleNotFoundException:
        print("⚠️ No medicine bottle detected")
    except Exception as err:
        print(f"❌ Error occurred: {str(err)}")
    finally:
        return coord, conf, image
            
def detect_with_ocr(ocr_model, image, coord_list: list[dict], tgt_label="ACE Inhibitor"):

    class LabelNotFoundException:
        pass

    wanted_rois = []

    try:
        for i, box in coord_list:
            xmin, ymin, xmax, ymax = map(int, box[:4])

            # 提取并放大ROI
            roi = image[ymin:ymax, xmin:xmax]


    except LabelNotFoundException:
        print("⚠️ No {} detected".format(tgt_label))
    except Exception as err:
        print(f"❌ Error occurred: {str(err)}")
    finally:
        return wanted_rois

    

def main(max_attempts=MAX_ATTEMPTS):

    # Initialise models
    im_model = YOLO("./Object_Detection/siou150.pt")
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Set language to English
    write_yaml_config(YAML_FPATH, YAML_CONFIG)

    attempted = 1
    roi_list = []

    while attempted <= max_attempts:
        print("[Attempt {}/{}] Start capturing in 5 seconds...".format(attempted, max_attempts))
        coord, conf, image = detect_medicine_exist(im_model)
        if coord == None and conf == None:
            attempted += 1
            continue
        roi_list = detect_with_ocr(ocr, coord, image)
        break    

    # After 3 attempts, no boxes are detected.
    if len(roi_list) == 0:
        print("\n❌ Failed 3 times. Please check:")
        print("1. Ensure bottle is centered in frame")
        print("2. Adjust camera focus")
        print("3. Clean camera lens")
        print("4. Improve lighting conditions")
        print("\n🔴 System stopped: Maximum error count reached")
    
    return roi_list

if __name__ == "__main__":
    main()