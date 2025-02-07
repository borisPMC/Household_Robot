from huggingface_hub import login
import torch
from transformers import pipeline
import Models
import Datasets

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device("cpu")
print("Using device:", DEVICE)

login()

# Global Hyper-param for controlled comparison
BATCH_SIZE = 16
NUM_WORKER = 2
EPOCH = 1
MAX_STEPS = 300

def build_dataset():
    Datasets.Custom_Dataset.build_new_dataset("borisPMC/grab_medicine_intent", "medicine_intent.csv")

def train_asr():
    
    token = {
        "en": "<|en|>",
        "yue": "<|yue|>",
    }

    ds = Datasets.CV17_dataset(["en", "yue"], token)

    whisper = Models.Whisper_Model(repo_id="borisPMC/whisper_grab_medicine_intent", use_exist=False, ds=ds)
    whisper.train()

def train_nlp():

    custom = Datasets.Custom_Dataset("borisPMC/grab_medicine_intent", True)
    nlp = Models.GPT2_Model("borisPMC/gpt2_grab_medicine_intent", False, custom)
    nlp.train()

def predict():
    pipe = pipeline("text-classification", model="borisPMC/bert_grab_medicine_intent", tokenizer="bert-base-multilingual-uncased")
    result = pipe("請攞甲福明畀我")
    print(result)

def main():
    train_asr()
    pass

if __name__ == "__main__":
    main()