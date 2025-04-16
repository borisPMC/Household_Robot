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

def auto_capture_photo(fpath, cam_index=0, quality=0.8):

    cv2.destroyAllWindows()

    # Open the default camera
    cam = cv2.VideoCapture(cam_index)

    if not cam.isOpened():
        print("Error: Could not access the camera.")
        return False

    print("Starting countdown for photo capture...")
    # Countdown timer for 3 seconds
    for seconds in range(3, 0, -1):
        print(f"Capturing in {seconds} seconds...")
        time.sleep(1)
    print("Capturing photo now!")
    
    # Capture a frame
    ret, frame = cam.read()
    if ret:
        # Save the captured frame as an image
        cv2.imwrite(fpath, frame)
        print(f"Photo saved to {fpath}")
    else:
        print("Failed to capture photo.")

    # Release the camera
    cam.release()
    cv2.destroyAllWindows()

    return fpath

def detect_medicine_exist(fname: str, im_model: YOLO) -> tuple[list[dict], list]:

    # Custom Exception for Expected Error (No bottle case)
    class BottleNotFoundException(Exception):
        pass

    # Initialise param
    conf = []
    coord = [dict] # Format for each item: {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

    try:
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
            # print(f"Bounding box: ({x_min}, {y_min}, {x_max}, {y_max})")
            # print(f"Confidence: {conf*100:.1f}%")
    
    except BottleNotFoundException:
        print("⚠️ No medicine bottle detected")
    finally:
        return coord, conf
            
def detect_with_ocr(ocr_model: PaddleOCR, image, coord_list: list[dict], tgt_label: str):

    # Sample input for coord_list: [<class 'dict'>, {'x_min': 237, 'y_min': 214, 'x_max': 293, 'y_max': 328}, {'x_min': 167, 'y_min': 186, 'x_max': 239, 'y_max': 329}]

    class LabelNotFoundException(Exception):
        pass

    wanted_rois = []
    tmp_dir = "./temp"

    try:
        # Start from 1: Skip class item
        for i in range(1, len(coord_list)):
            xmin = coord_list[i]["x_min"]
            ymin = coord_list[i]["y_min"]
            xmax = coord_list[i]["x_max"]
            ymax = coord_list[i]["y_max"]

            # 提取并放大ROI
            roi = image[ymin:ymax, xmin:xmax]
            resized_roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # 保存临时文件
            output_path = "./temp/roi_{}.jpg".format(i)
            cv2.imwrite(output_path, resized_roi)

            # OCR识别
            ocr_result = ocr_model.ocr(output_path, det=True, rec=True)

            if ocr_result[0] == None:
                raise LabelNotFoundException

            # 检查目标文本
            text_found = False
            detected_text = ocr_result[0][0][1][0]
            if tgt_label.lower().replace("_", " ") == detected_text.lower():
                text_found = True

            # 只记录包含目标文本的ROI
            if text_found:
                roi_info = {
                    "id": i,
                    "coordinates": (xmin, ymin, xmax, ymax),
                    "center": ((xmin + xmax)//2, (ymin + ymax)//2),
                    "ocr_text": tgt_label  # 记录匹配到的文本
                }
                wanted_rois.append(roi_info)
                break

        if len(wanted_rois) == 0:
            raise LabelNotFoundException
        
    except LabelNotFoundException:
        print("⚠️ No {} detected".format(tgt_label))

    except Exception:
        print("Unexpected Error!")
    
    finally:
        return wanted_rois



# Main function for the master program
def detect_medicine(model_dict, target_label: str, max_attempts=MAX_ATTEMPTS) -> list[dict]:

    detect_med_model = model_dict["detect_med_model"]
    ocr_model = model_dict["ocr_model"]
    camera_index = model_dict["medicine_camera_index"]

    print("Detecting medicine:", target_label, "\n")

    snapshot_fname = './temp/photo.jpg'
    coord_list = [] # Perhaps List of Dicts
    attempted = 1
    roi_list = [] # {id, coordinates (xmin, ymin, xmax, ymax), center, ocr_text}
    coord = []
    conf = []

    for i in range(attempted, max_attempts+1):

        # try:
            # Step 1: Taking a snapshot
            print("[Attempt {}/{}] Start capturing in few seconds...".format(i, max_attempts))
            auto_capture_photo(snapshot_fname)
            image = cv2.imread(snapshot_fname)
            
            # Step 2: See if ANY medicine exists
            coord, conf = detect_medicine_exist(image, detect_med_model)

            # Step 3: See if the specific medicine is found
            roi_list = detect_with_ocr(ocr_model, image, coord, target_label)

            # Success to find specific medicine(s), exit the loop
            for tgt in roi_list:
                coord_list.append({
                    "xmin": tgt["coordinates"][0],
                    "ymin": tgt["coordinates"][1],
                    "xmax": tgt["coordinates"][2],
                    "ymax": tgt["coordinates"][3],
                })
            
            if len(coord_list) > 0:
                break
        
        # except Exception:
        #     print("Error!")
        


    # After 3 attempts, no boxes are detected.
    if len(coord_list) == 0:
        print("\n❌ Failed 3 times. Please check:")
        print("1. Ensure bottle is centered in frame")
        print("2. Adjust camera focus")
        print("3. Clean camera lens")
        print("4. Improve lighting conditions")
        print("\n🔴 System stopped: Maximum error count reached")
    else:
        print("Medicine found!\n")
    
    return coord_list

def main():
    detect_medicine()


if __name__ == "__main__":
    main()