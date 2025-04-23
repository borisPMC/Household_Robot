import os
import cv2
import torch
import time
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

class PoseEstimator_ViTPose:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models()
        self.person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        self.person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=self.device)
        self.pose_image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        self.pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=self.device)

    """
    Download if models not 
    """
    def load_models(self):

        person_repo = "PekingU/rtdetr_r50vd_coco_o365"
        pose_repo = "usyd-community/vitpose-base-simple"
        
        # Load the models for pose estimation and person detection
        if os.path.exists(f"./temp/{person_repo}"):
            self.person_image_processor = AutoProcessor.from_pretrained(f"./temp/{person_repo}")
            self.person_model = RTDetrForObjectDetection.from_pretrained(f"./temp/{person_repo}", device_map=self.device)
        
        else:
            print("Downloading Person Model...")
            self.person_image_processor = AutoProcessor.from_pretrained(person_repo)
            self.person_model = RTDetrForObjectDetection.from_pretrained(person_repo).to(self.device)
            self.person_model.save_pretrained(f"./temp/{person_repo}")
            self.person_image_processor.save_pretrained(f"./temp/{person_repo}")

        if os.path.exists(f"./temp/{pose_repo}"):
            self.pose_image_processor = AutoProcessor.from_pretrained(f"./temp/{pose_repo}")
            self.pose_model = VitPoseForPoseEstimation.from_pretrained(f"./temp/{pose_repo}", device_map=self.device)
        
        else:
            print("Downloading Pose Model...")
            self.pose_image_processor = AutoProcessor.from_pretrained(pose_repo)
            self.pose_model = VitPoseForPoseEstimation.from_pretrained(pose_repo).to(self.device)
            self.pose_model.save_pretrained(f"./temp/{pose_repo}")
            self.pose_image_processor.save_pretrained(f"./temp/{pose_repo}")

        self.pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(self.device)
        self.person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(self.device)

    # For the load_image method, it does not have to input an url, as long as it returns an image object
    def load_image(self, fpath):
        return Image.open(fpath)

    def detect_humans(self, image):
        inputs = self.person_image_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.person_model(**inputs)
        results = self.person_image_processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
        )
        result = results[0]  # take first image results
        person_boxes = result["boxes"][result["labels"] == 0].cpu().numpy()
        person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
        person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
        return person_boxes

    def detect_keypoints(self, image, person_boxes):
        inputs = self.pose_image_processor(image, boxes=[person_boxes], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.pose_model(**inputs)
        pose_results = self.pose_image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
        return pose_results[0]  # results for first image

    def estimate_pose(self, fpath: str):
        image = self.load_image(fpath)
        person_boxes = self.detect_humans(image)
        keypoints = self.detect_keypoints(image, person_boxes)
        return keypoints

def is_detected(keypoints, confidence_threshold=0.5):
    # Check if the list is empty
    if not keypoints:
        return False

    # Extract the first dictionary in the list
    keypoint_data = keypoints[0]

    # Check if 'keypoints' tensor is empty
    if keypoint_data['keypoints'].numel() == 0:
        return False

    # Check if any keypoint has a confidence score above the threshold
    scores = keypoint_data['scores']
    if scores.numel() > 0 and (scores > 0.5).any():
        return True

    return False

# Altered version for live capture to reduce latency (Used Image instead of Frame)
def live_capture(pose_estimator: PoseEstimator_ViTPose, device=0, img_temp_fpath="./temp/su_temp.jpg", confidence_threshold=0.5, timer=5):

    # Live capture a footage. 0: default camera
    cam = cv2.VideoCapture(device)

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_fpath, fourcc, 20.0, (frame_width, frame_height))

    # Define returning var
    keypoints = []          # List[list]: List of 2 scalar (x, y) lists
    user_exist = False

    start_time = time.time()
    while True:
        ret, frame = cam.read()

        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # # Display the captured frame
        # cv2.imshow('Camera', frame)
        cv2.imwrite(img_temp_fpath, frame)

        # Check if 5 second is passed or a person is detected
        keypoints = pose_estimator.estimate_pose(img_temp_fpath)
        user_exist = is_detected(keypoints, confidence_threshold)

        if time.time() - start_time >= timer or user_exist:
            break
    
    cam.release()
    # out.release()
    cv2.destroyAllWindows()
    
    return keypoints, user_exist

# Main function for the Master program
# Expected to be run forever
def find_user_thread(model_dict, shared_dict, listen_event) -> None:

    pose_model = model_dict["pose_model"]
    device = model_dict["user_camera_index"]
    print(f"SU Thread: Using Camera {device}. Start finding user...")

    while True:

        listen_event.wait()

        # # Idle when grabbing medicine
        # if shared_dict["user_flag"] and shared_dict["current_cmd"]:
        #     # print("SU Thread: Idle")
        #     time.sleep(5) 
        #     continue

        # Simulate finding a user
        # time.sleep(5)  # Simulate the 5-second duration for finding a user
        shared_dict["keypoints"], shared_dict["user_flag"] = live_capture(pose_model, device, timer=shared_dict["THREAD_PROCESS_TIMER"])

        # Wait 1 second before looping again
        time.sleep(2)
