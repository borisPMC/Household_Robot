import transformers
import datasets
import pandas as pd
from preprocessing import hybrid_split

def getColsForTrain(example):

    intent = example["intent"]
    speech = example["sentence"]
    tkn = ["0"] * len(hybrid_split(speech.lower()))
    ner_tag = "'" + "".join(tkn)

    result = {
        "Speech": speech,
        "Intent": intent,
        "NER_Tag": ner_tag,
    }

    return result


def import_jsonl(jsonl_path, train_data_file="train.jsonl", test_data_file="test.jsonl"):
    """Import a jsonl file and return a list of dictionaries."""
    ds_train = datasets.load_dataset(jsonl_path, data_files=train_data_file, split="train")
    filtered_train = datasets.Dataset.from_dict({
        "intent": ds_train["intent"],
        "sentence": ds_train["sentence"]
    })
    ds_test = datasets.load_dataset(jsonl_path, data_files=test_data_file, split="train")
    filter_test = datasets.Dataset.from_dict({
        "intent": ds_test["intent"],
        "sentence": ds_test["sentence"]
    })
    mapped_train = filtered_train.map(getColsForTrain, remove_columns=["sentence", "intent"])
    mapped_test = filter_test.map(getColsForTrain, remove_columns=["sentence", "intent"])
    ds = datasets.DatasetDict(
        {
            "train": mapped_train,
            "test": mapped_test,
        }
    )
    return ds

slurp_en = import_jsonl("Intent_Prediction/training_data/slurp/")

print(slurp_en["train"][36])

import asyncio
from googletrans import Translator

test_set = slurp_en["train"][:100]["Speech"]

async def translate_text():
    async with Translator() as translator:
        translations = await translator.translate(test_set, dest='Cantonese')
        for translation in translations:
            print(translation.origin, ' -> ', translation.text)

asyncio.run(translate_text())