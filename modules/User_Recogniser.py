import cv2, torch, time, serial, threading
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

class PoseEstimator_ViTPose:
    def __init__(self, device=None, person_repo = "PekingU/rtdetr_r50vd_coco_o365", pose_repo = "usyd-community/vitpose-base-simple"):

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", cache_dir=f"./temp/{person_repo}")
        self.person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", cache_dir=f"./temp/{person_repo}", device_map=self.device)
        self.pose_image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple", cache_dir=f"./temp/{pose_repo}")
        self.pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", cache_dir=f"./temp/{person_repo}", device_map=self.device)

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

class MIC_array:
    def __init__(self, COM, states_dict, listen_event):
        self.ser_ = serial.Serial(port=COM, baudrate=115200)#, parity=serial.PARITY_ODD,stopbits=serial.STOPBITS_TWO,bytesize=serial.SEVENBITS)
        thread = threading.Thread(target=self.reader(), kwargs={"states_dict": states_dict, "listen_event": listen_event})
        thread.start()
        self.now_pos = -1
        print("voice direction detector:", self.ser_.isOpen)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Must add this magic method to prevent the port not closing and obstract the next initialization.
        """
        self.ser_.close()

    def __delattr__(self):
        self.ser_.close()

    def reader(self, states_dict, listen_event):
        while True:
            listen_event.wait()
            if self.ser_.in_waiting > 0:
                data  = self.ser_.readline()
                datastr = data.decode('utf-8')
                self.now_pos = float(datastr.strip())
                self.update_state(states_dict)
                #print(self.now_pos)
                time.sleep(0.01)

    def get_pos(self):
        return self.now_pos
    
    def update_state(self, states_dict):
        radian = (self.get_pos() / 180) if (self.get_pos() > 0 and self.get_pos() <= 180) else 0
        states_dict["user_angle"] = radian
    
    def close(self):
        """
        Close the serial port. Should be called whenever the program ends.
        """
        self.ser_.close()
        return

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
    if scores.numel() > 0 and (scores > confidence_threshold).any():
        return True

    return False

def live_capture(pose_estimator: PoseEstimator_ViTPose, device=1, img_temp_fpath="./temp/su_temp.jpg", confidence_threshold=0.5, timer=1):

    # Live capture a footage. 0: default camera
    cam = cv2.VideoCapture(device)

    # Define returning var
    keypoints = []          # List[list]: List of 2 scalar (x, y) lists
    user_exist = False

    start_time = time.time()
    while True:
        try:
            ret, frame = cam.read()

            if not ret:
                print("Failed to capture frame. Exiting.")
                break

            # # Display the captured frame
            cv2.imwrite(img_temp_fpath, frame)

            # Check if 5 second is passed or a person is detected
            keypoints = pose_estimator.estimate_pose(img_temp_fpath)
            user_exist = is_detected(keypoints, confidence_threshold)

            if time.time() - start_time >= timer or user_exist:
                break
        except ValueError:
            keypoints = []
            user_exist = False
            break
    
    cam.release()
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
        shared_dict["keypoints"], shared_dict["user_flag"] = live_capture(pose_model, device, timer=shared_dict["THREAD_PROCESS_TIMER"])

        time.sleep(1)

# PoseEstimator_ViTPose()