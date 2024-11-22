from ultralytics import YOLO
import torch

pose_model = YOLO("yolo11n-pose.pt")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_frame(f):
    f.save()

def get_video_pose_feature(url: str) -> list:

    # Run batched inference on a list of images
    video_frames = pose_model.predict(
        source=url,
        conf=0.5,
        device="cuda:0",
        max_det=1,
        stream=True,
        verbose=False,
    )  # return a list of Results objects

    frame_list = []
    # img_width, img_height = video_frames[0].orig_shape

    for f in video_frames:
        keypoint_list = []
        for kpt in f.keypoints:
            if kpt.conf == None:
                # Zero padding
                # print("None detected")
                frame_list.append([[0] * 3 for _ in range(17)])
            else:
                coord = kpt[0].xyn.flatten(0, 1)
                conf = kpt[0].conf.flatten()
                # Normalised coordination
                for i in range(17):
                    x = coord[i][0].item()
                    y = coord[i][1].item()
                    c = conf[i].item()
                    data = [x, y, c]
                    keypoint_list.append(data)
        frame_list.append(keypoint_list) # 17 keypoints for YOLOv11
        # f.show()
    
    return frame_list