# Get Charades from HuggingFace

from datasets import ( # v2.22.0 Note: using v3 produces fsspec.exceptions.FSTimeoutError.
    load_dataset, DownloadConfig, Dataset, BuilderConfig
  )
import torch
from backbone import get_video_pose_feature
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", DEVICE)

def download_dataset(save_dir, datasets_name):

    dlConfig = DownloadConfig(
        cache_dir = save_dir,
        resume_download=True,
        max_retries=100
    )

    # Load the dataset
    dataset = load_dataset(
        path = datasets_name,
        download_config=dlConfig,
        trust_remote_code=True,
    )

    return dataset

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
    ds = ds.map(convert_to_tensor)
    return ds

ACTION_DATA_LIST = {
    "train": [],
    "test": [],
}

def build_action_class_dataset(ds: Dataset):

    ds_train = ds["train"].select(range(1))
    ds_test = ds["test"].select(range(1))

    global ACTION_DATA_LIST

    def add_action_feature(example, split="train"):

        frame_list = get_video_pose_feature(example["video"])
        fps = len(frame_list) / example["length"]

        for i in range(len(example["labels"])):
            action = example["labels"][i]
            s_frame = int(example["action_timings"][i][0] * fps) # starting frame
            f_frame = int(example["action_timings"][i][1] * fps) # finishing frame
            features = frame_list[s_frame:f_frame]
            ACTION_DATA_LIST[split].append({
                "action_class": action,
                "pose_features": features,
            })

        return example

    ds_train = ds_train.map(add_action_feature, fn_kwargs={"split": "train"})
    ds_test = ds_test.map(add_action_feature, fn_kwargs={"split": "test"})

    ACTION_TRAIN_DS = Dataset.from_list(ACTION_DATA_LIST["train"], split="train")
    ACTION_TEST_DS = Dataset.from_list(ACTION_DATA_LIST["test"], split="test")

    ACTION_TRAIN_DS.save_to_disk("./action_ds")
    ACTION_TEST_DS.save_to_disk("./action_ds")

    print("Done saving")