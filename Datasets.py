from typing import List, Optional, Union
from datasets import load_dataset, Dataset, interleave_datasets

SEED = 10000

class Custom_Dataset:

    repo_id: str = "borisPMC/grab_medicine_intent"
    dataset: dict

    def __init__(self, repo_id,  use_exist: bool, config_list=["English", "Cantonese"]):
        self.repo_id = "borisPMC/grab_medicine_intent"
        self._set_metadata()
        if use_exist:
            self._call_dataset_workflow(self.repo_id, config_list)

    def _call_dataset_workflow(self, repo, config_list: List[str]):
        # Basic logic calling datasets

        self.datasets = {}
        for config in config_list:
            self.datasets[config] = self._call_dataset(repo, config)
    
    def _call_dataset(self, repo: str, config: str) -> Dataset:
        return load_dataset(repo, name=config)

    def _set_metadata(self):
        self.input_col = "Speech"
        self.output_col = "Label"
        self.class_list = ["Empty", "ACE Inhibitor", "Metformin", "Atorvastatin", "Amitriptyline"]

    def group_train_test(self):
        
        train_ds = interleave_datasets([ds["train"] for ds in self.datasets.values()], probabilities=[0.5,0.5], seed=SEED)
        test_ds = interleave_datasets([ds["test"] for ds in self.datasets.values()], probabilities=[0.5,0.5], seed=SEED)

        return train_ds, test_ds

    @staticmethod
    def  build_new_dataset(repo_id, csv_path):
        ds = load_dataset("csv", data_files=csv_path, split="train")
        splited_ds = ds.train_test_split(0.3, shuffle=True, seed=1, writer_batch_size=16)

        for l in ["Cantonese", "English"]:

            pushing_ds = splited_ds.filter(lambda example: example["Language"] == l)
            pushing_ds = pushing_ds.remove_columns("Language")

            print(pushing_ds)

            pushing_ds.push_to_hub(
                repo_id=repo_id,
                config_name=l,
                private=True
            )

class CV17_dataset:

    def __init__(self, lang_list, token: Optional[dict]=None):
        self.lang_list = lang_list

        self.train_ds = self._get_compiled_ds("train", token)
        self.test_ds = self._get_compiled_ds("test", token)

        self.input_col = "audio"
        self.output_col = "sentence"

    def _get_compiled_ds(self, split: str, token: Optional[dict]=None):
        
        sub_ds_list = []
        for l in self.lang_list:
            sub_ds_list.append(CV17_dataset._load_ds(l, split, token))

        ds = interleave_datasets(sub_ds_list, seed=SEED, stopping_strategy="first_exhausted")
        ds = ds.remove_columns(["client_id", "path", "gender", "accent", "segment", "age", 'up_votes', 'down_votes', 'locale', 'variant'])
        return ds

    @staticmethod
    def _load_ds(lang, split, lang_token: Optional[dict]=None):
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            lang,
            split=split,
            trust_remote_code=True,
            streaming=True,
        )

        # Add language token to the dataset
        if lang_token:
            ds = ds.map(lambda x: {"sentence": (lang_token[lang] + x["sentence"]), "audio": x["audio"]})

        return ds

# def main():
#     Custom_Dataset.build_new_dataset("grab_medicine_intent", "C:/Users/20051248d/Desktop/fyp_git/newer_data.csv")

# if __name__ == "__main__":
#     main()   