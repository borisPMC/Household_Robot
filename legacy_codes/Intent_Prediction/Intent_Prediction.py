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
from transformers import pipeline, Pipeline, BertConfig, WhisperForConditionalGeneration, WhisperProcessor, AutoModel
from multitask_BERT_for_hf import Multitask_BERT_v2
import Models
import Datasets
import sounddevice as sd

import asyncio
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

def build_dataset():
    Datasets.PharmaIntent_Dataset.build_new_dataset("borisPMC/PharmaIntent", "medicine_intent.csv")

def train_asr(ds: Datasets.New_PharmaIntent_Dataset, ds_config: dict, output_repo: str, pretrain_model: str) -> None:

    use_exist = False

    whisper = Models.Whisper_Model(
        repo_id=output_repo,
        pretrain_model=pretrain_model,
        use_exist=use_exist
    )

    trainer = None
    
    if ds_config["merge_language"] == False:
        for l in ds_config["languages"]:
            ds.set_splits_by_lang(l)
            trainer = whisper.update_trainer(ds, trainer)
            trainer.train()
    else:
        trainer = whisper.update_trainer(ds, trainer)
        trainer.train()

    # Push the best model (lowest WER) model to HF
    print(f"Pushing final model to hub...")
    trainer.push_to_hub(f"{whisper.repo_id}")

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

def evaluate_unseen(prediction: dict, label: dict) -> float:

    evaluators = {
        "f1_metric": evaluate.load("f1"),
        "seq_f1_metric": evaluate.load("seqeval")
    }

    intent_f1 = evaluators["f1_metric"].compute(predictions=prediction["intent"], references=label["intent"], average="macro")["f1"]
    med_seq_f1 = evaluators["seq_f1_metric"].compute(predictions=prediction["ner"], references=label["ner"])["overall_f1"]

    print(
        f"Intent F1: {intent_f1:.4f} | Medicine List F1: {med_seq_f1:.4f}"
    )

    return

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

    audio_list = ds.test_ds["Audio"]
    intent_list = [str(Datasets.New_PharmaIntent_Dataset.INTENT_LABEL.index(i)) for i in ds.test_ds["Intent"]]
    ner_list = [Datasets.New_PharmaIntent_Dataset.check_NER(i) for i in ds.test_ds["NER_Labels"]]

    true_labels = {
        "intent": intent_list,
        "ner":  ner_list,
    }

    ner_repo = nlp_repo + "_ner"
    intent_repo = nlp_repo + "_intent"

    asr_pipe = pipeline("automatic-speech-recognition", model=asr_repo)
    asr_pipe.generation_config.forced_decoder_ids = None
    ner_pipe = pipeline("token-classification", model=ner_repo)
    intent_pipe = pipeline("text-classification", model=intent_repo)

    transcripts = [i["text"] for i in asr_pipe(audio_list)]
    print(transcripts)
    output = {
        "transcript": transcripts,
        "intent": [i["label"][-1] for i in intent_pipe(transcripts)],
        "ner": [Datasets.New_PharmaIntent_Dataset.post_process_med(i) for i in ner_pipe(transcripts)],
    }

    # raise Exception(f"{output}")

    evaluate_unseen(output, true_labels)
    return

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

def compare_models(ds, model):

    model.generation_config.forced_decoder_ids = None

    audio_list = ds.test_ds["Audio"]
    intent_list = [str(Datasets.New_PharmaIntent_Dataset.INTENT_LABEL.index(i)) for i in ds.test_ds["Intent"]]
    ner_list = [Datasets.New_PharmaIntent_Dataset.check_NER(i) for i in ds.test_ds["NER_Labels"]]

    true_labels = {
        "intent": intent_list,
        "ner":  ner_list,
    }

    nlp_repo = "borisPMC/MedicGrabber_multitask_BERT"

    ner_repo = nlp_repo + "_ner"
    intent_repo = nlp_repo + "_intent"

    ner_pipe = pipeline("token-classification", model=ner_repo)
    intent_pipe = pipeline("text-classification", model=intent_repo)

    transcripts = [i["text"] for i in model(audio_list)]
    output = {
        "transcript": transcripts,
        "intent": [i["label"][-1] for i in intent_pipe(transcripts)],
        "ner": [Datasets.New_PharmaIntent_Dataset.post_process_med(i) for i in ner_pipe(transcripts)],
    }
    evaluate_unseen(output, true_labels)
    return

def main():
    
    # ds_config = {
    #         "use_exist": True,
    #         "languages": ["Cantonese", "English"],
    #         "merge_language": True,
    #     }
    # ds = Datasets.call_dataset(ds_config)

    # train_asr(ds, ds_config, "borisPMC/MedicGrabber_WhisperTiny", "openai/whisper-tiny")
    # train_asr(ds, ds_config, "borisPMC/MedicGrabber_WhisperSmall", "openai/whisper-small")
    # train_asr(ds, ds_config, "borisPMC/MedicGrabber_WhisperLargeTurbo", "openai/whisper-large-v3-turbo")

    # test_manual_nlp("borisPMC/whisper_small_grab_medicine_intent")
    
    ds = Datasets.call_dataset({
            "use_exist": True,
            "languages": ["Cantonese", "English"],
            "merge_language": True,
        })
    
    print(ds.train_ds[0])
    
    # test_ds(ds, "borisPMC/MedicGrabber_WhisperTiny", "borisPMC/MedicGrabber_multitask_BERT")          10 epo: Intent F1: 0.5576 | Medicine List F1: 0.9789
    test_ds(ds, "borisPMC/MedicGrabber_WhisperSmall", "borisPMC/MedicGrabber_multitask_BERT")
    # test_ds(ds, "borisPMC/MedicGrabber_WhisperLargeTurbo", "borisPMC/MedicGrabber_multitask_BERT")

    # hf_model = .from_pretrained("./temp/multitask_BERT_MedicGrabber/checkpoint-170")

    # hf_model.push_to_hub("borisPMC/multitask_BERT_MedicGrabber", commit_message="Uploading Multitask BERT model")
    # tokenizer.push_to_hub("borisPMC/multitask_BERT_MedicGrabber")

if __name__ == "__main__":
    main()