import data
import datasets
import pandas as pd
# from transformers import ViTImageProcessor

# from ultralytics import YOLO

# # Row: Video data;           Label: video_id, video, subject, scene, quality, relevance, verified, script, objects[], descriptions[], labels[], action_timings[][2], length
# ds = data.download_dataset("./charades", "HuggingFaceM4/charades")
# data.build_action_class_dataset(ds)

# Row: each action sequence; Label: action_class, pose_feature [F][x, y, c] (Possible to extend and dense for NN...?)

ds = data.load_from_local()
print(ds)

# print(ds_train_action[0])
# print(ds_test_action[0])