# from cv2 import compare
from huggingface_hub import login
import numpy as np
from pandas import read_csv
import torch
from tqdm import tqdm
from transformers import pipeline
import Models
import Datasets
import sounddevice as sd

import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device("cpu")
print("Using device:", DEVICE)

login("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")

def build_dataset():
    Datasets.MedIntent_Dataset.build_new_dataset("borisPMC/grab_medicine_intent", "medicine_intent.csv")

def train_asr():
    
    ds = Datasets.MedIntent_Dataset(True)
    use_exist = os.path.exists("borisPMC/whisper_small_grab_medicine_intent")

    for l in ["English", "Cantonese", "Eng_Can" ,"Can_Eng"]:
        whisper = Models.Whisper_Model(
            repo_id="borisPMC/whisper_small_grab_medicine_intent",
            pretrain_model="openai/whisper-small",
            use_exist=use_exist, 
            dataset=ds.filter(lambda example: example["Language"] == l)
        )
        whisper.train()

    # whisper = Models.Whisper_Model(
    #     repo_id="borisPMC/whisper_small_grab_medicine_intent",
    #     pretrain_model="openai/whisper-small",
    #     use_exist=use_exist, 
    #     dataset=ds)
    # whisper.train()

def train_nlp():

    custom = Datasets.MedIntent_Dataset(True)
    nlp = Models.Whisper_Model("borisPMC/gpt2_grab_medicine_intent", False, custom)
    nlp.train()

def predict_asr(audiopath):

    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    result = pipe(audiopath)
    return result

def predict_nlp(transcript):

    pipe = pipeline("text-classification", model="borisPMC/bert_grab_medicine_intent", tokenizer="bert-base-multilingual-uncased")
    result = pipe(transcript)
    return result

def evaluate(prediction: list[str], label: list[str]) -> float:

    print("Predicted:", prediction)
    print("Label:", label)

    if len(prediction) != len(label):
        raise ValueError("Prediction and label length mismatch")
    
    correct = 0
    for i in range(len(prediction)):
        correct += (prediction[i] == label[i])

    return correct / len(prediction)

# Simulates deployment on the robot
def live_test(duration=5, sample_rate=16000) -> str:

    print("Recording for 5 seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()  # Wait until recording is finished
    audio_array = np.squeeze(audio_data)  # Convert to 1D array

    transcript = predict_asr(audio_array)
    class_label = predict_nlp(transcript)

    print("Transcript:", transcript)
    print("Class Label:", class_label)

    return class_label["label"]

def random_test(n) -> None:

    prediction = []
    confidence = []

    df = read_csv("medicine_intent.csv", nrows=n)
    label_list = df["Label"].tolist()
    audiopath_list = df["Audio_path"].tolist()

    for path in tqdm(audiopath_list):
        transcript = predict_asr(path)
        class_label = predict_nlp(transcript)
        prediction.append(class_label["label"])
        confidence.append(class_label["score"])

    print(evaluate(prediction, label_list))

def main():

    build_dataset()

if __name__ == "__main__":
    main()