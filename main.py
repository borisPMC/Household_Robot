import data
import datasets
import pandas as pd
# from transformers import ViTImageProcessor

# from ultralytics import YOLO

# Row: Video data;           Label: video_id, video, subject, scene, quality, r  elevance, verified, script, objects[], descriptions[], labels[], action_timings[][2], length
ds = data.download_dataset("./charades", "HuggingFaceM4/charades") # Run this line first
data.build_action_class_dataset(ds, 50, "./action_ds_2")

# Row: each action sequence; Label: action_class, pose_feature [F][x, y, c] (Possible to extend and dense for NN...?)

# ds = data.load_from_local("./action_ds_1")
# print(ds.shape)
# print(ds.filter(lambda example: example['split'] == "train"))
# print(ds.filter(lambda example: example['split'] == "test"))

# print(ds_train_action[0])
# print(ds_test_action[0])