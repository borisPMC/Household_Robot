# Get Charades from HuggingFace

from datasets import ( # v2.22.0 Note: using v3 produces fsspec.exceptions.FSTimeoutError.
    load_dataset, DownloadConfig, Dataset, BuilderConfig, concatenate_datasets, DatasetDict
  )
import torch
from backbone import get_video_pose_feature
import numpy as np
import pandas as pd
from tqdm import tqdm

from math import (floor, ceil)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", DEVICE)

def download_dataset(save_dir, datasets_name):

    dlConfig = DownloadConfig(
        cache_dir = save_dir,
        # resume_download=True,
        force_download=True,
        max_retries=100
    )

    # Load the dataset
    dataset = load_dataset(
        path = datasets_name,
        data_dir = save_dir,
        download_config=dlConfig,
        trust_remote_code=True,
    )

    return dataset

# RegEx incase random arrow file names

MAP_DIR = "charades/extracted/c5d9fa5ce5c9f33b8cd4b12d4344107f12d99fc384fd061c0c3f0bb758bd8932/Charades/Charades_v1_mapping.txt"
CLASSNAME_DIR = "charades/extracted/c5d9fa5ce5c9f33b8cd4b12d4344107f12d99fc384fd061c0c3f0bb758bd8932/Charades/Charades_v1_verbclasses.txt"

def get_class_map(map_dir = MAP_DIR, name_dir = CLASSNAME_DIR) -> pd.DataFrame:

    # Each item: ID, Lable, Org_Class
    VERB_LIST = []
    item_list = []

    with open(name_dir, "r") as cf:
        for line in cf:
            item_list = line.split(sep=" ")
            VERB_LIST.append({
                "vID": item_list[0],
                "Label": item_list[1][:-1], # remove \n at the end
            })

    VERB_DF = pd.DataFrame(VERB_LIST)

    MAP_LIST = []

    with open(map_dir, "r") as mf:
        for line in mf:
            item_list = line.split(sep=" ")
            MAP_LIST.append({
                "cID": item_list[0],
                "vID": item_list[2][:-1],
            })

    MAP_DF = pd.DataFrame(MAP_LIST)
    # MAP_DF = MAP_DF.set_index("cID")

    RETURN_DF = pd.merge(MAP_DF, VERB_DF, "left", on="vID")
    RETURN_DF.set_index("cID")

    return RETURN_DF

def load_from_local(path = "./action_ds"):

    """
    Output:
    Dataset(
        action_class: str(cXXX)
        pose_feature: Tensor([frames][51])" <- padded 0s
    )
    """

    def convert_to_tensor(example):

        org_pose_feature = example["pose_features"]

        for i in range(len(example["pose_features"])):
            if org_pose_feature[i] == []:
                example["pose_features"][i] = [0] * 51 # padding zeros
            else:
                # Flatten each frame
                example["pose_features"][i] = [j for kp in org_pose_feature[i] for j in kp]

        # Convert to tensor
        example["pose_features"] = torch.tensor(example["pose_features"], dtype=torch.float32)
        return example

    ds = Dataset.load_from_disk(path)
    ds = ds.map(convert_to_tensor, batched=True, batch_size=500, cache_file_name=".load_cache")
    return ds

def build_action_class_dataset(org_ds: Dataset, sameple_size:int, save_dir: str):

    train_test_ratio = 0.7

    selected_ds = concatenate_datasets([
        org_ds["train"].take(ceil(sameple_size * train_test_ratio)),
        org_ds["test"].take(floor(sameple_size * (1-train_test_ratio)))
    ])

    ACTION_DS = Dataset.from_dict({
        "action_class": [],
        "pose_features": [],
    })

    action_label_map = get_class_map()

    iter = selected_ds.iter(batch_size=64)

    for batch in tqdm(iter):

        # List of actions per video
        action_dict = {
                "action_class": [],
                "pose_features": [],
        }

        for e in range(len(batch["video"])):

            if batch["video"][e] == None: break

            frame_list = get_video_pose_feature(batch["video"][e])
            fps = len(frame_list) / batch["length"][e]

            for i in range(len(batch["labels"][e])):
                action = batch["labels"][e][i]
                s_frame = int(batch["action_timings"][e][i][0] * fps) # starting frame
                f_frame = int(batch["action_timings"][e][i][1] * fps) # finishing frame
                selected_frames = frame_list[s_frame:f_frame]
                # Flatten the kpt data per frame
                flatten_frames = []
                for i in range(len(selected_frames)):
                    f = [val for kpt in selected_frames[i] for val in kpt]
                    flatten_frames.append(f)

                # Append extracted clip data.
                action_dict["action_class"].append(action_label_map.at[action, "Label"])
                action_dict["pose_features"].append(flatten_frames)
        
        concat_ds = Dataset.from_dict(action_dict)

        ACTION_DS = concatenate_datasets([ACTION_DS, concat_ds])

    ACTION_DS.save_to_disk(save_dir)
    print(ACTION_DS.shape)
    print(ACTION_DS.data)

    print("Done saving")