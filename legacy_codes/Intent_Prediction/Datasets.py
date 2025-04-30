import os
import re
from typing import List, Optional, Union
from datasets import load_dataset, Dataset, interleave_datasets, Audio, DatasetDict

from collections import Counter
import json
import numpy as np
from pandas import read_excel

SEED = 42

class PharmaIntent_Dataset:

    train_ds: Dataset
    test_ds: Dataset
    valid_ds: Dataset

    def __init__(self, use_exist: bool, group_by_lang: bool, config_list=["English", "Cantonese", "Eng_Can" ,"Can_Eng"]):
        self._set_metadata()
        if use_exist:
            self._call_dataset_workflow(self.repo_id, config_list, group_by_lang)

    def _call_dataset_workflow(self, repo, config_list: List[str], group_by_lang: bool):

        # Call each language dataset, then merge them and split into train and test
        self.datasets = {}
        
        for config in config_list:
                self.datasets[config] = load_dataset(repo, name=config)

        if (group_by_lang == False):
            self.train_ds, self.test_ds, self.valid_ds = self._group_train_test()

        print("Dataset loaded successfully! \n")
    

    def _set_metadata(self):
        self.repo_id = "borisPMC/PharmaIntent"
        self.audio_col = "Audio"
        self.speech_col = "Text"
        self.label_col = "Label"
        self.label_list = ["Empty", "ACE Inhibitor", "Metformin", "Atorvastatin", "Amitriptyline"]

    def _group_train_test(self):
        
        train_ds = interleave_datasets([ds["train"] for ds in self.datasets.values()], stopping_strategy="all_exhausted")
        test_ds = interleave_datasets([ds["test"] for ds in self.datasets.values()], stopping_strategy="all_exhausted")
        valid_ds = interleave_datasets([ds["validation"] for ds in self.datasets.values()], stopping_strategy="all_exhausted")

        return train_ds, test_ds, valid_ds
    
    def create_vocab_file(self, output_dir: str, vocab_filename: str = "vocab.json"):
        """
        Creates a vocabulary file from the sentences in the dataset.

        Args:
            output_dir (str): Directory to save the vocabulary file.
            vocab_filename (str): Name of the vocabulary file (default: "vocab.json").
        """

        

        # Combine all sentences from train, validation, and test datasets
        print("Extracting sentences from the dataset...")
        all_sentences = (
            self.train_ds["Text"] +
            self.valid_ds["Text"] +
            self.test_ds["Text"]
        )

        # Count unique characters
        print("Counting unique characters...")
        counter = Counter("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijiklmnopqrstuvwxyz".join(all_sentences))
        
        unique_chars = sorted(counter.keys())

        # Create vocabulary dictionary
        print("Creating vocabulary dictionary...")
        vocab = {char: idx for idx, char in enumerate(unique_chars)}


        # add special tokens
        vocab["[UNK]"] = len(vocab)
        vocab["[PAD]"] = len(vocab)
        vocab["|"] = vocab[" "]
        del vocab[" "]

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save vocabulary to file
        vocab_path = os.path.join(output_dir, vocab_filename)
        print(f"Saving vocabulary to {vocab_path}...")
        with open(vocab_path, "w", encoding="utf-8") as vocab_file:
            json.dump(vocab, vocab_file, ensure_ascii=False, indent=4)

        print("Vocabulary file created successfully!")

    # # Truncate/pad audio to 5 seconds (5 * 16000 samples for 16kHz sampling rate) for fair model comparison
    # @staticmethod
    # def preprocess_audio(example):
    #     max_length = 5 * 16000  # 5 seconds at 16kHz
    #     audio_array = example["Audio"]["array"]
        
    #     # Truncate if longer than 5 seconds
    #     if len(audio_array) > max_length:
    #         audio_array = audio_array[:max_length]
    #     # Pad with zeros if shorter than 5 seconds
    #     elif len(audio_array) < max_length:
    #         padding = np.zeros(max_length - len(audio_array), dtype=audio_array.dtype)
    #         audio_array = np.concatenate([audio_array, padding])
        
    #     example["Audio"]["array"] = audio_array
    #     return example

    @staticmethod
    def  build_new_dataset(repo_id, csv_path):
        ds = load_dataset("csv", data_files=csv_path, split="train")

        # Load Audio into dataset
        ds = ds.cast_column("Audio_path", Audio(sampling_rate=16000))
        ds = ds.rename_column("Audio_path", "Audio")

        ds = ds.map(PharmaIntent_Dataset.preprocess_audio)

        total_amt = len(ds)

        train_amt = int(total_amt * 0.8 - (total_amt * 0.8 % 16))
        test_amt = int(total_amt - train_amt)

        print(ds)

        splited_ds = ds.train_test_split(test_size=test_amt, shuffle=True, seed=SEED)
        feed_ds = splited_ds["train"].train_test_split(test_size=0.2, shuffle=True, seed=SEED)

        train_ds = feed_ds["train"]
        validate_ds = feed_ds["test"]
        test_ds = splited_ds["test"]

        doneDS = DatasetDict({
            "train": train_ds,
            "validation": validate_ds,
            "test": test_ds
        })

        # English: 118 train, 27 valid, 34 test -> 179
        # Cantonese: 114 train, 30 valid, 40 test -> 184
        # Eng_Can: 40 train, 6 valid, 14 test -> 60
        # Can_Eng: 35 train, 14 valid, 11 test -> 60

        # Can_Eng: Cantonese as main, English as secondary
        # Eng_Can: English as main, Cantonese as secondary

        for l in ["English", "Cantonese", "Eng_Can" ,"Can_Eng"]:

            pushing_ds = doneDS.filter(lambda example: example["Language"] == l)
            pushing_ds = pushing_ds.remove_columns("Language")

            print("Pushing config: " + l + "\n")
            print(pushing_ds)

            pushing_ds.push_to_hub(
                repo_id=repo_id,
                config_name=l,
                private=True
            )

class IntentDataset:

    INTENT_LABEL = [
        "enquire_info",
        "retrieve",
        "enquire_location",
        "enquire_suitable_med",
        "general_chat",
        "set_furniture",
        "set_software"
    ]
    # O: irelevant B: beginning I: inside
    NER_LABEL = ["O", "B-ACE_Inhibitor", "I-ACE_Inhibitor", "B-Metformin", "I-Metformin", "B-Atorvastatin", "I-Atorvastatin", "B-Amitriptyline", "I-Amitriptyline",]

    datasets: dict[DatasetDict]
    train_ds: Dataset
    test_ds: Dataset
    valid_ds: Dataset

    def __init__(self, repo_id: str, config: dict):
        self.repo_id = repo_id

        self._set_metadata()
        self._call_dataset_workflow(config) if config["use_exist"] else None

    def _call_dataset_workflow(self, config:dict):

        # Call each language dataset, then merge them and split into train and test
        self.datasets = {}
        
        for lang in config["languages"]:
            lang_ds = load_dataset(self.repo_id, name=lang)
            processed_lang_ds = lang_ds.map(IntentDataset.postdownload_process)
            self.datasets[lang] = processed_lang_ds

        # Merge the datasets and set split with the merged one
        if (config["merge_language"]):
            self.group_lang()

        print("Dataset loaded successfully! \n")
    

    def _set_metadata(self):
        self.audio_col = "Audio_Path"
        self.speech_col = "Text"
        self.intent = "Intent"
        self.ner_tag = "NER_Tag"

    def set_splits_by_lang(self, lang):
        self.train_ds = self.datasets[lang]["train"]
        self.valid_ds = self.datasets[lang]["valid"]
        self.test_ds = self.datasets[lang]["test"]

        print(f"{lang} dataset loaded!")
        print("Train:", len(self.train_ds), "Valid:", len(self.valid_ds), "Test:", len(self.test_ds))
        return

    def group_lang(self):
        self.train_ds = interleave_datasets([ds["train"] for ds in self.datasets.values()], stopping_strategy="all_exhausted")
        self.test_ds = interleave_datasets([ds["test"] for ds in self.datasets.values()], stopping_strategy="all_exhausted")
        self.valid_ds = interleave_datasets([ds["valid"] for ds in self.datasets.values()], stopping_strategy="all_exhausted")

        print(f"Merged dataset loaded!")
        print("Train:", len(self.train_ds), "Valid:", len(self.valid_ds), "Test:", len(self.test_ds))
        return

    # Use for validation / deployment, return 4-item list (padded)
    @staticmethod
    def check_NER(NER_Tag: Union[str, list[str]], fill_na=True) -> list[str]:
        
        if type(NER_Tag) == str:
            tag_list = list(NER_Tag)
        else:
            tag_list = NER_Tag
        
        detect_med = set()

        for token in tag_list:
            if token != "0" and IntentDataset.NER_LABEL[int(token)][2:] != "":
                detect_med.add(IntentDataset.NER_LABEL[int(token)][2:]) # [2:] -> remove the beginning and interim tag

        listed_med = list(detect_med)

        if fill_na:
            listed_med = listed_med + ["Empty"] * (4 - len(listed_med))

        return listed_med

    @staticmethod
    def post_process_med(output: list[dict]) -> list[4]:
        """
        Processing output list generated by NER Pipeline. Return a list of medicine lists
        """
        ner_tag = []
        for tkn in output:
            ner_tag.append(tkn["entity"][-1])
        detect_med = IntentDataset.check_NER(ner_tag)
        return detect_med

    @staticmethod
    def postdownload_process(example):

        tokenized_speech = []
        ner_labels = []
        speech = example["Text"]

        # Preprocess NER_Tag
        if example["NER_Tag"][0] == "'":
            ner_tag = example["NER_Tag"][1:]
        else:
            ner_tag = example["NER_Tag"]

        # Preprocess Text
        tokens = hybrid_split(speech)
        tokenized_speech = tokens

        # Preprocess Intent
        num_intent = IntentDataset.INTENT_LABEL.index(example["Intent"])
        
        # Ensure NER_Tag length matches the number of tokens
        if len(ner_tag) != len(tokens):
            raise ValueError(f"Mismatch between tokens and NER_Tag: {speech}")

        ner_labels = [int(tag) for tag in ner_tag]
        
        example["Tokenized_Speech"] = tokenized_speech
        example["NER_Labels"] = ner_labels
        example["Intent_Label"] = num_intent

        return example

    @staticmethod
    def  build_new_dataset(repo_id, config):
        ds = load_dataset("csv", data_files=config["data_file"], split="train")

        # Load Audio into dataset
        ds = ds.cast_column("file_name", Audio(sampling_rate=16000))
        ds = ds.rename_column("file_name", "Audio")

        # ds = ds.map(PharmaIntent_Dataset.preprocess_audio)

        total_amt = len(ds)

        train_amt = int(total_amt * config["train_ratio"] - (total_amt * config["train_ratio"] % 16))
        test_amt = int(total_amt - train_amt)

        splited_ds = ds.train_test_split(test_size=test_amt, shuffle=True, seed=SEED)
        feed_ds = splited_ds["train"].train_test_split(test_size=1-config["train_ratio"], shuffle=True, seed=SEED)

        train_ds = feed_ds["train"]
        validate_ds = feed_ds["test"]
        test_ds = splited_ds["test"]

        doneDS = DatasetDict({
            "train": train_ds,
            "valid": validate_ds,
            "test": test_ds
        })

        for l in ["English", "Cantonese"]:

            pushing_ds = doneDS.filter(lambda example: example["Language"] == l)
            pushing_ds = pushing_ds.remove_columns("Language")

            print("Pushing config: " + l + "\n")
            print(pushing_ds)

            pushing_ds.push_to_hub(
                repo_id=repo_id,
                config_name=l,
                private=True
            )


class CV17_dataset:
    
    datasets: dict[Union[DatasetDict, Dataset]]
    train_ds: Dataset
    test_ds: Dataset
    valid_ds: Dataset

    def __init__(self, lang_list, token: Optional[dict]=None):
        self.lang_list = lang_list

        self.train_ds = self._get_compiled_ds("train", token)
        self.test_ds = self._get_compiled_ds("test", token)

        self.input_col = "audio"
        self.output_col = "sentence"

    def _get_compiled_ds(self, split: str, token: Optional[dict]=None):
        
        self.datasets = {}
        for l in self.lang_list:
            self.datasets[l] = CV17_dataset._load_ds(l, split, token)

        # ds = interleave_datasets(sub_ds_list, seed=SEED, stopping_strategy="first_exhausted")
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
    
    def create_vocab_file(self, fpath):
        pass

def hybrid_split(string: str) -> List[str]:
    """
    Split a string into tokens using a hybrid regex.

    Args:
        string (str): Input string.

    Returns:
        List[str]: List of tokens.
    """
    regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
    matches = re.findall(regex, string, re.UNICODE)
    return matches

def build_dataset():
    IntentDataset.build_new_dataset(
        "PharmaIntent_v2", 
        config={
            "data_file": "temp\ds.csv", 
            "lang": ["Cantonese", "English"],
            "train_ratio": 0.8,
        })

def convert_excel_to_csv(rpath, wpath):
    ds = read_excel(rpath, dtype=str, index_col=None)
    ds.to_csv(wpath, index=False)

def call_dataset(config=None):

    if not config:
        print("Applying default config for calling PharmaIntent_v2...")
        config = {
            "use_exist": True,
            "languages": ["Cantonese", "English"],
            "merge_language": False,
        }

    ds = IntentDataset("borisPMC/PharmaIntent_v2", config)

    print("Dataset loaded successfully! \n")
    return ds

def main():
    # convert_excel_to_csv("Intent_Prediction\multitask_audio\multitask_ds_light.xlsx", "temp\ds.csv")
    build_dataset()

if __name__ == "__main__":
    main()   