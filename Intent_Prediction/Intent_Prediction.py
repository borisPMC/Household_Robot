"""

Main function ONLY FOR INTENT PREDICTION

"""

# from cv2 import compare
from time import sleep
import evaluate
from huggingface_hub import login
import numpy as np
from pandas import read_csv
import torch
from tqdm import tqdm
from transformers import pipeline, Pipeline
import Models
import Datasets
import sounddevice as sd

import asyncio
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

login("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")

def build_dataset():
    Datasets.PharmaIntent_Dataset.build_new_dataset("borisPMC/PharmaIntent", "medicine_intent.csv")

def train_asr(ds: Datasets.New_PharmaIntent_Dataset, output_repo: str, pretrain_model: str) -> None:

    use_exist = False

    whisper = Models.Whisper_Model(
        repo_id=output_repo,
        pretrain_model=pretrain_model,
        use_exist=use_exist
        )
    
    whisper.train(ds)
    return

    # Uncomment this if you want to train on the entire dataset
    # whisper = Models.Whisper_Model(
    #     repo_id="borisPMC/whisper_small_grab_medicine_intent",
    #     pretrain_model="openai/whisper-small",
    #     use_exist=use_exist, 
    #     dataset=ds)
    # whisper.train()

def train_nlp(output_repo: str, pretrain_model: str) -> None:

    ds = Datasets.PharmaIntent_Dataset(True, group_by_lang=False)

    for l in ["English", "Cantonese", "Eng_Can" ,"Can_Eng"]:

        ds.train_ds = ds.datasets[l]["train"]
        ds.valid_ds = ds.datasets[l]["validation"]
        ds.test_ds = ds.datasets[l]["test"]

        print("Train:", len(ds.train_ds), "Valid:", len(ds.valid_ds), "Test:", len(ds.test_ds))

        use_exist = os.path.exists(output_repo)
        nlp = Models.BERT_Model(
            repo_id=output_repo,
            pretrain_model=pretrain_model,
            use_exist=use_exist, 
            dataset=ds)
        
        nlp.train()

def evaluate_unseen(prediction: list[str], label: list[str]) -> float:

    evaluators = {
        "f1_metric": evaluate.load("f1"),
        "seq_f1_metric": evaluate.load("seqeval")
    }

    print("Predicted:", prediction)
    print("Label:", label)

    if len(prediction) != len(label):
        raise ValueError("Prediction and label length mismatch")
    
    correct = 0
    for i in range(len(prediction)):
        correct += (prediction[i] == label[i])

    return correct / len(prediction)

# Simulates deployment on the robot
def listen_audio(asr_pipe: Pipeline, nlp_pipe: Pipeline, duration=5, sample_rate=16000) -> str:

    print("Recording for 5 seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()  # Wait until recording is finished
    audio_array = np.squeeze(audio_data)  # Convert to 1D array

    transcript = asr_pipe(audio_array)
    class_label = nlp_pipe(transcript)

    # print("Transcript:", transcript)
    print("Class Label:", class_label)

    return class_label["label"]

def test_ds(ds: Datasets.New_PharmaIntent_Dataset, asr_repo: str, nlp_repo: str) -> None:

    prediction = []
    confidence = []

    # df = read_csv("medicine_intent.csv", nrows=n)
    # label_list = df["Label"].tolist()
    # audiopath_list = df["Audio_path"].tolist()

    audio_list = ds.test_ds["Audio"]
    intent_list = ds.test_ds["Intent"]
    ner_list = [Datasets.New_PharmaIntent_Dataset.check_NER(i) for i in ds.test_ds["NER_Labels"]]

    true_labels = {
        "intent": intent_list,
        "ner":  ner_list,
    }

    asr_pipe = pipeline("automatic-speech-recognition", model=asr_repo)
    asr_pipe.generation_config.forced_decoder_ids = None
    nlp_pipe = pipeline("feature-extraction", model=nlp_repo)

    transcript = asr_pipe(audio_list[0])
    output = nlp_pipe(transcript["text"])

    raise Exception(f"{transcript} | {len(output[0][0])}")

    for audio in tqdm(audio_list):
        transcript = asr_pipe(audio)
        class_label = nlp_pipe(transcript)
        prediction.append(class_label["label"])
        confidence.append(class_label["score"])

    print(evaluate_unseen(prediction, label_list))

    raise Exception(f"{audio_list} | {intent_list} | {ner_list}")

    asr_pipe = pipeline("automatic-speech-recognition", model=asr_repo)
    asr_pipe.generation_config.forced_decoder_ids = None
    nlp_pipe = pipeline("text-classification", model=nlp_repo, tokenizer="bert-base-multilingual-uncased")

    for audio in tqdm(audio_list):
        transcript = asr_pipe(audio)
        class_label = nlp_pipe(transcript)
        prediction.append(class_label["label"])
        confidence.append(class_label["score"])

    print(evaluate_unseen(prediction, label_list))

def test_manual_nlp(asr_repo: str) -> None:

    ds = Datasets.PharmaIntent_Dataset(True, group_by_lang=False)

    prediction = []
    confidence = []

    # df = read_csv("medicine_intent.csv", nrows=n)
    # label_list = df["Label"].tolist()
    # audiopath_list = df["Audio_path"].tolist()

    label_list = ds.test_ds["Label"]
    audio_list = ds.test_ds["Audio"]

    asr_pipe = pipeline("automatic-speech-recognition", model=asr_repo)
    asr_pipe.generation_config.forced_decoder_ids = None
    classifier = Models.TableSearcher()

    for audio in tqdm(audio_list):
        transcript = asr_pipe(audio)["text"]
        classifier.predict(transcript)
        classifier.generate_response()

        class_label = classifier.pop_medicine()

        if class_label is None or class_label == "":
            class_label = "Empty"

        # Hard reset for testing datasets
        classifier._reset_state()

        prediction.append(class_label)
        confidence.append(1.0)

    print(evaluate_unseen(prediction, label_list))

def main():
    
    # build_dataset()

    # ds = Datasets.call_dataset()

    # train_asr(ds, "borisPMC/MedicGrabber_WhisperTiny", "openai/whisper-tiny")
    # train_asr(ds, "borisPMC/MedicGrabber_WhisperSmall", "openai/whisper-small")
    # train_asr("borisPMC/whisper_large_grab_medicine_intent", "openai/whisper-large-v3")
    # train_asr("borisPMC/whisper_largeTurbo_grab_medicine_intent", "openai/whisper-large-v3-turbo")

    # train_asr("borisPMC/xlsr_grab_medicine_intent", "facebook/wav2vec2-large-xlsr-53")
    # train_nlp("borisPMC/bert_baseM_grab_medicine_intent", "bert-base-multilingual-uncased")

    # test_manual_nlp("borisPMC/whisper_small_grab_medicine_intent")
    
    ds = Datasets.call_dataset({
            "use_exist": True,
            "languages": ["Cantonese", "English"],
            "merge_language": True,
        })
    
    test_ds(ds, "borisPMC/MedicGrabber_WhisperTiny", "borisPMC/multitask_BERT_MedicGrabber")
    test_ds(ds, "borisPMC/MedicGrabber_WhisperSmall", "borisPMC/multitask_BERT_MedicGrabber")
    # test_ds("borisPMC/whisper_large_grab_medicine_intent", "borisPMC/bert_grab_medicine_intent")
    # test_ds("openai/whisper-large-v3-turbo", "borisPMC/bert_grab_medicine_intent")

    return

if __name__ == "__main__":
    main()