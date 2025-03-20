from typing import List, Optional, Union
from datasets import load_dataset, Dataset, interleave_datasets, Audio

import numpy as np

SEED = 42

class MedIntent_Dataset:

    def __init__(self, use_exist: bool, config_list=["English", "Cantonese"]):
        self._set_metadata()
        if use_exist:
            self._call_dataset_workflow(self.repo_id, config_list)

    def _call_dataset_workflow(self, repo, config_list: List[str]):

        # Call each language dataset, then merge them and split into train and test
        self.datasets = {}
        for config in config_list:
            self.datasets[config] = load_dataset(repo, name=config)
        
        self.train_ds, self.test_ds = self._group_train_test()
        print("Dataset loaded successfully ", self.train_ds, self.test_ds)
    

    def _set_metadata(self):
        self.repo_id = "borisPMC/grab_medicine_intent"
        self.audio_col = "Audio"
        self.speech_col = "Speech"
        self.label_col = "Label"
        self.label_list = ["Empty", "ACE Inhibitor", "Metformin", "Atorvastatin", "Amitriptyline"]

    def _group_train_test(self):
        
        train_ds = interleave_datasets([ds["train"] for ds in self.datasets.values()], stopping_strategy="all_exhausted")
        test_ds = interleave_datasets([ds["test"] for ds in self.datasets.values()], stopping_strategy="all_exhausted")

        return train_ds, test_ds
    
    # Truncate/pad audio to 5 seconds (5 * 16000 samples for 16kHz sampling rate) for fair model comparison
    @staticmethod
    def preprocess_audio(example):
        max_length = 5 * 16000  # 5 seconds at 16kHz
        audio_array = example["Audio"]["array"]
        
        # Truncate if longer than 5 seconds
        if len(audio_array) > max_length:
            audio_array = audio_array[:max_length]
        # Pad with zeros if shorter than 5 seconds
        elif len(audio_array) < max_length:
            padding = np.zeros(max_length - len(audio_array), dtype=audio_array.dtype)
            audio_array = np.concatenate([audio_array, padding])
        
        example["Audio"]["array"] = audio_array
        return example

    @staticmethod
    def  build_new_dataset(repo_id, csv_path):
        ds = load_dataset("csv", data_files=csv_path, split="train")

        # Load Audio into dataset
        ds = ds.cast_column("Audio_path", Audio(sampling_rate=16000))
        ds = ds.rename_column("Audio_path", "Audio")

        ds = ds.map(MedIntent_Dataset.preprocess_audio)

        splited_ds = ds.train_test_split(0.2, shuffle=True, seed=1, writer_batch_size=16)

        for l in ["Cantonese", "English"]:

            pushing_ds = splited_ds.filter(lambda example: example["Language"] == l)
            pushing_ds = pushing_ds.remove_columns("Language")

            print(pushing_ds)

            pushing_ds.push_to_hub(
                repo_id=repo_id,
                config_name=l,
                private=True
            )

# class CV17_dataset:

#     def __init__(self, lang_list, token: Optional[dict]=None):
#         self.lang_list = lang_list

#         self.train_ds = self._get_compiled_ds("train", token)
#         self.test_ds = self._get_compiled_ds("test", token)

#         self.input_col = "audio"
#         self.output_col = "sentence"

#     def _get_compiled_ds(self, split: str, token: Optional[dict]=None):
        
#         sub_ds_list = []
#         for l in self.lang_list:
#             sub_ds_list.append(CV17_dataset._load_ds(l, split, token))

#         ds = interleave_datasets(sub_ds_list, seed=SEED, stopping_strategy="first_exhausted")
#         ds = ds.remove_columns(["client_id", "path", "gender", "accent", "segment", "age", 'up_votes', 'down_votes', 'locale', 'variant'])
#         return ds

#     @staticmethod
#     def _load_ds(lang, split, lang_token: Optional[dict]=None):
#         ds = load_dataset(
#             "mozilla-foundation/common_voice_17_0",
#             lang,
#             split=split,
#             trust_remote_code=True,
#             streaming=True,
#         )

#         # Add language token to the dataset
#         if lang_token:
#             ds = ds.map(lambda x: {"sentence": (lang_token[lang] + x["sentence"]), "audio": x["audio"]})

#         return ds

def main():
    # MedIntent_Dataset.build_new_dataset("grab_medicine_intent", "medicine_intent.csv")
    ds_obj = MedIntent_Dataset(use_exist=True)
    print(ds_obj.train_ds[0]['Audio']['array'].shape)

if __name__ == "__main__":
    main()   