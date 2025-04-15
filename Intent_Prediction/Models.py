"""


This script is for storing self-defined model classes, NOT FOR CALLING DUE TO CUDA ALLOCATION PROBLEM ATM!


"""
# SET GLOBAL, DONT CHANGE
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 4
MAX_STEPS = 300
EVAL_STEPS = 30

LLM_EPOCH = 3
ASR_EPOCH = 10

SEED = 42

from dataclasses import dataclass
import os
import re
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor, WhisperConfig,
    BertForSequenceClassification, BertTokenizer, BertConfig,
    Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, 
    GPT2ForSequenceClassification, GPT2Tokenizer, GPT2Config,
    Trainer, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset, IterableDataset, Audio
from Datasets import PharmaIntent_Dataset
import evaluate

from Intent_Prediction import Datasets

# List of testing Whisper models

# Source: https://huggingface.co/openai/whisper-tiny
# Params: 39M (tiny)

# Source: https://huggingface.co/openai/whisper-small
# Params: 244M (small)

# Source: https://huggingface.co/openai/whisper-large-v3
# Params: 1550 M (large-3)

# Source: https://huggingface.co/openai/whisper-large-v3-turbo
# Params: 809 M (large-v3-turbo)

# Custom Repo-id: borisPMC/whisper_[]_grab_medicine_intent

class Whisper_Model:

    repo_id: str
    model: WhisperForConditionalGeneration
    # feature_extractor: WhisperFeatureExtractor
    # tokenizer: WhisperTokenizer

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        # feature_extractor: Union[Any, WhisperFeatureExtractor]
        # tokenizer: Union[Any, WhisperTokenizer]
        # decoder_start_token_id: int

        def __init__(self, processor: WhisperProcessor, audio_pad_id=0, text_pad_id=50257, audio_max_length=480000, text_max_length=128):
            self.processor = processor
            self.audio_pad_id = audio_pad_id
            self.text_pad_id = text_pad_id
            self.audio_max_length = audio_max_length
            self.text_max_length = text_max_length
            pass

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [
                {"input_features": feature["input_features"][0]} for feature in features
            ]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch

    def __init__(self, repo_id: str, pretrain_model: str, use_exist: bool):
        self.repo_id = repo_id

        # Use existing model or load pre-trained Whisper model
        call_id = repo_id if use_exist & (repo_id != None) else pretrain_model

        self._import_model_set(call_id)
        self._prepare_training()

    def _import_model_set(self, call_id):

        print("Importing model:", call_id)

        self.model = WhisperForConditionalGeneration.from_pretrained(call_id)
        self.processor = WhisperProcessor.from_pretrained(call_id, task="transcribe")
        self.model.generation_config.forced_decoder_ids = None

    def _prepare_datasets(self, ds: Datasets.New_PharmaIntent_Dataset):
        # Cast audio column to 16kHz sampling rate
        train_ds = ds.train_ds.cast_column("Audio", Audio(sampling_rate=16000))
        valid_ds = ds.valid_ds.cast_column("Audio", Audio(sampling_rate=16000))

        def preprocess_function(example):
            audio = example["Audio"]

            example = self.processor(
                audio=audio["array"],
                sampling_rate=audio["sampling_rate"],
                text=example["Speech"],
            )

            # compute input length of audio sample in seconds
            example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

            return example

        # Apply preprocessing
        processed_train_ds = train_ds.map(preprocess_function)
        processed_valid_ds = valid_ds.map(preprocess_function)

        return processed_train_ds, processed_valid_ds

    def _prepare_training(self):
        self.evaluator = evaluate.load("wer")

        self.data_collator = self.DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
        )

        self.training_args = Seq2SeqTrainingArguments(
            output_dir=f"./temp/{self.repo_id}",
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=VAL_BATCH_SIZE,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            predict_with_generate=True,
            seed=SEED,
            num_train_epochs=ASR_EPOCH,
            fp16=True,
            fp16_full_eval=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            metric_for_best_model="wer_ortho",
            greater_is_better=False,
            remove_unused_columns=False,
        )

        return

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        # compute orthographic wer
        wer_ortho = 100 * self.evaluator.compute(predictions=pred_str, references=label_str)

        # # # compute normalised WER
        # pred_str_norm = [self.processor(pred) for pred in pred_str]
        # label_str_norm = [self.processor(label) for label in label_str]
        # # filtering step to only evaluate the samples that correspond to non-zero references:
        # pred_str_norm = [
        #     pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
        # ]
        # label_str_norm = [
        #     label_str_norm[i]
        #     for i in range(len(label_str_norm))
        #     if len(label_str_norm[i]) > 0
        # ]

        # wer = 100 * self.evaluator.compute(predictions=pred_str_norm, references=label_str_norm)

        return {"wer_ortho": wer_ortho}

    def train(self, ds: Datasets.New_PharmaIntent_Dataset):

        ds.group_lang()

        train_ds, valid_ds = self._prepare_datasets(ds)

        # for l in ds.datasets.keys():

        #     ds.set_splits_by_lang(l)

        #     train_ds, valid_ds = self._prepare_datasets(ds)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            processing_class=self.processor,
        )

        if not train_ds or not valid_ds:
            print("Missing Dataset(s)!")
            return
        
        trainer.train()
        
        # model_id = self.repo_id.split("/")[1]
        # trainer.save_model(f"./temp/{model_id}_lang_{l}_checkpoint")

class BERT_Model:
    def __init__(self, repo_id: str, pretrain_model: str, use_exist: bool, dataset: PharmaIntent_Dataset):
        self.model_id = repo_id
        self.dataset = dataset
        self.label_list = dataset.label_list

        # Use existing model or load pre-trained BERT model
        call_id = repo_id if use_exist & (repo_id != None) else pretrain_model

        self._set_processor(pretrain_model)
        self._call_model(call_id)
        self._preprocess_dataset()
        self._prepare_training()

    def _set_processor(self, model_id: str):
        self.tokenizer = BertTokenizer.from_pretrained(model_id)

    def _call_model(self, model_id: str):
        id2label = {i: label for i, label in enumerate(self.label_list)}
        label2id = {label: i for i, label in enumerate(self.label_list)}

        config = BertConfig(
            id2label=id2label,
            label2id=label2id,
            num_labels=len(self.label_list),
        )

        self.model = BertForSequenceClassification.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)
        self._resize_token_embeddings()

    def _resize_token_embeddings(self):
        if self.tokenizer and self.model:
            vocab_size = len(self.tokenizer)
            self.model.resize_token_embeddings(vocab_size)

    def _preprocess_dataset(self):
        def preprocess_function(batch):
            speech = batch["Speech"]
            label = batch["Label"]

            # Tokenize speech and map labels
            tokenized = self.tokenizer(speech, padding="max_length", truncation=True, return_tensors="pt")
            batch["input_ids"] = tokenized["input_ids"]
            batch["attention_mask"] = tokenized["attention_mask"]
            batch["label"] = [self.label_list.index(l) for l in label]
            return batch

        # Apply preprocessing
        self.training_dataset = self.dataset.train_ds.map(preprocess_function, remove_columns=["Speech", "Label", "Audio"], batched=True)
        self.testing_dataset = self.dataset.valid_ds.map(preprocess_function, remove_columns=["Speech", "Label", "Audio"], batched=True)

    def _prepare_training(self):
        self.evaluator = evaluate.load("f1")
        self.data_collator = DataCollatorWithPadding(self.tokenizer, return_tensors="pt")

        training_args = TrainingArguments(
            output_dir=f"./{self.model_id}",
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=VAL_BATCH_SIZE,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            num_train_epochs=LLM_EPOCH,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            dataloader_drop_last=True,
            push_to_hub=True,
            hub_private_repo=True,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=self.training_dataset,
            eval_dataset=self.testing_dataset,
            compute_metrics=self._compute_metrics,
        )

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.evaluator.compute(predictions=predictions, references=labels, average="weighted")

    def train(self):
        if not self.training_dataset or not self.testing_dataset:
            print("Missing Dataset(s)!")
            return
        self.trainer.train()

# List of testing Wav2Vec2 models

# Source: https://huggingface.co/facebook/wav2vec2-base-960h
# Params: 94M (base-960h) # English-specific

# Source: https://huggingface.co/facebook/wav2vec2-large-960h
# Params: 317M (large-960h) # English-specific

# Source: https://huggingface.co/facebook/wav2vec2-large-xlsr-53
# Params: 317M (large-xlsr-53) # Multilingual

# Custom Repo-id: borisPMC/wav2vec2_[]_grab_medicine_intent

class Wav2Vec2_Model:

    model_id: str
    model: Wav2Vec2ForCTC
    processor: Wav2Vec2Processor
    feature_extractor: Wav2Vec2FeatureExtractor
    tokenizer: Wav2Vec2CTCTokenizer

    @dataclass
    class DataCollatorForW2V2:
        """
        Data collator for Wav2Vec2 models. Handles padding for both input features and labels.
        """
        processor: Union[Any, Wav2Vec2Processor]
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            """
            Pads input features and labels to the same length and prepares them for training.

            Args:
                features (List[Dict[str, Union[List[int], torch.Tensor]]]): List of features containing input values and labels.

            Returns:
                Dict[str, torch.Tensor]: A dictionary containing padded input features and labels.
            """
            # Separate input features and labels
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            # Pad input features
            batch = self.processor.feature_extractor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )

            # Pad labels
            with self.processor.as_target_processor():
                labels_batch = self.processor.tokenizer.pad(
                    label_features,
                    padding=self.padding,
                    return_tensors="pt",
                )

            # Replace padding token with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # Add labels to the batch
            batch["labels"] = labels

            return batch

    def __init__(self, repo_id: str, pretrain_model: str, use_exist: bool, dataset: PharmaIntent_Dataset):
        self.model_id = repo_id
        self.dataset = dataset

        # Use existing model or load pre-trained Whisper model
        call_id = repo_id if use_exist & (repo_id != None) else pretrain_model

        self._import_model_set(call_id, repo_id)
        self._prepare_datasets()
        self._prepare_training()

    def _import_model_set(self, call_id, repo_id):
        self.model = Wav2Vec2ForCTC.from_pretrained(call_id, ignore_mismatched_sizes=True)

        try:
            # Try to load the processor from the Hugging Face repository
            self.processor = Wav2Vec2Processor.from_pretrained(repo_id)
            print("Processor loaded successfully from Hugging Face repository.")
        except Exception as e:
            print(f"Processor not found in repository. Creating a new processor. Error: {e}")

            # Create a new processor using a custom vocabulary file
            vocab_path = "./vocab/vocab.json"  # Path to the custom vocabulary file
            if not os.path.exists(vocab_path):
                raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}. Please create it first.")

            self.processor = Wav2Vec2_Model.create_processor(fpath=vocab_path)
            print("Processor created successfully using the custom vocabulary file.")

            # Save the processor locally and push it to the Hugging Face Hub
            self.processor.save_pretrained(repo_id, push_to_hub=False)
            print(f"Processor saved and pushed to Hugging Face Hub at {repo_id}.")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(repo_id)
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(repo_id)

        self.model.config.vocab_size = len(self.tokenizer)

    def _prepare_datasets(self):
        # Cast audio column to 16kHz sampling rate
        self.dataset.train_ds = self.dataset.train_ds.cast_column("Audio", Audio(sampling_rate=16000))
        self.dataset.valid_ds = self.dataset.valid_ds.cast_column("Audio", Audio(sampling_rate=16000))

        def preprocess_function(batch):
            audio = batch["Audio"]
            label = batch["Speech"]

            # Ensure audio is truncated/padded to 5 seconds (5 * 16000 samples)
            max_length = 5 * 16000  # 5 seconds at 16kHz
            audio_array_list = [a["array"][:max_length] if len(a["array"]) > max_length else np.pad(
            a["array"], (0, max_length - len(a["array"])), mode="constant") for a in audio]

            # batched output is "un-batched" to ensure mapping is correct
            batch["input_values"] = [self.processor(arr, sampling_rate=16000).input_values[0] for arr in audio_array_list] 
            
            with self.processor.as_target_processor():
                batch["labels"] = self.processor(label).input_ids
            return batch


        # Apply preprocessing
        self.train_dataset = self.dataset.train_ds.map(preprocess_function, remove_columns=["Audio", "Speech", "Label"], batched=True)
        self.test_dataset = self.dataset.valid_ds.map(preprocess_function, remove_columns=["Audio", "Speech", "Label"], batched=True)

    def _prepare_training(self):
        self.cer_metric = evaluate.load("cer")

        self.data_collator = self.DataCollatorForW2V2(
            processor=self.processor,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./{self.model_id}",
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=VAL_BATCH_SIZE,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            fp16=True,
            predict_with_generate=True,
            generation_max_length=225,
            seed=SEED,
            num_train_epochs=ASR_EPOCH,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            dataloader_drop_last=True,
            metric_for_best_model="CER",
            greater_is_better=False,
            push_to_hub=True,
            hub_private_repo=True,
        )


        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            data_collator=self.data_collator,
            tokenizer=self.processor.feature_extractor,
            compute_metrics=self.compute_metrics,
        )

    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions.argmax(-1)

        # Decode predictions and labels
        pred_str = self.processor.batch_decode(pred_ids)
        labels_ids[labels_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(labels_ids, group_tokens=False)

        # Compute CER
        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
        return {"CER": cer}
    
    @staticmethod
    def create_processor(fpath: str):
        
        # Tokenizer vocab is custom from MedicIntent.create_vocab_file()
        tokenizer = Wav2Vec2CTCTokenizer(
            fpath, 
            unk_token="[UNK]", 
            pad_token="[PAD]", 
            word_delimiter_token="|",
            model_max_length=1024)
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        return processor

    def train(self):
        if not self.train_dataset or not self.test_dataset:
            print("Missing Dataset(s)!")
            return
        self.trainer.train()

        # Save processor
        self.processor.save_pretrained(self.model_id, push_to_hub=True)


class GPT2_Model:

    model_id: str
    model: GPT2ForSequenceClassification
    tokenizer: GPT2Tokenizer

    training_dataset: Union[Dataset, IterableDataset]
    testing_dataset: Union[Dataset, IterableDataset]

    datacollator: DataCollatorWithPadding
    evaluator: evaluate.EvaluationModule
    trainer: Trainer

    label_list: List[str]

    def __init__(self, model_id: str, use_exist: bool, dataset_class: PharmaIntent_Dataset):
        self.model_id = model_id
        self.label_list = dataset_class.label_list

        if use_exist:
            call_id = self.model_id
        else:
            call_id = "openai-community/gpt2"

        self._set_processor(call_id)
        self._call_model(call_id)
        self._preprocess_dataset(dataset_class)

        # Define first avoid CUDA error
        self._prepare_training()

    def _set_processor(self, model_id: str):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.pad_token = self.tokenizer.eos_token

    def _call_model(self, model_id: str):

        # Set labels for the model

        id2label = {}
        label2id = {}

        for i in range(len(self.label_list)):
            id2label[i] = self.label_list[i]
            label2id[self.label_list[i]] = i
        
        config = GPT2Config(
            id2label = id2label,
            label2id = label2id,
            num_labels = 5,
            pad_token_id = 50256
        )
        
        self.model = GPT2ForSequenceClassification.from_pretrained(model_id, config=config)
        self._resize_token_embeddings()
    
    # Ensure the model's token embeddings match the tokenizer's vocabulary size (due default: 2 classes | this ds: 5 classes)
    def _resize_token_embeddings(self):
        if self.tokenizer and self.model:
            vocab_size = len(self.tokenizer)
            self.model.resize_token_embeddings(vocab_size)

    def _preprocess_dataset(self, dataset: PharmaIntent_Dataset):

        # No language tokens for BERT

        train_ds, valid_ds = dataset.group_train_test()

        def _f(batch):

            speech_col = batch["Speech"]
            label_col = batch["Label"]

            tokenised_dict = self.tokenizer(speech_col, padding="max_length", return_tensors="pt")
            # output_ids = self.tokenizer(label_col, padding="max_length", return_tensors="pt").input_ids

            batch["input_ids"] = tokenised_dict.input_ids
            batch["attention_mask"] = tokenised_dict.attention_mask
            batch["label"] = [self.label_list.index(l) for l in label_col]

            return batch

        vectorized_train_ds = train_ds.map(_f, batched=True, remove_columns=["Label","Speech"], batch_size=16)
        vectorized_test_ds = valid_ds.map(_f, batched=True, remove_columns=["Label","Speech"] , batch_size=16)

        self.training_dataset = vectorized_train_ds
        self.testing_dataset = vectorized_test_ds

    def _compute_metrics(self, eval_pred):
        print(eval_pred)
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        print(predictions, labels)
        return self.evaluator.compute(predictions=predictions, references=labels, average="weighted")

    def _prepare_training(self):
        self.evaluator = evaluate.load("f1")
        self.datacollator = DataCollatorWithPadding(self.tokenizer, return_tensors="pt")
        training_args = TrainingArguments(
            output_dir=("./"+self.model_id),  # change to a repo name of your choice
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=VAL_BATCH_SIZE,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size Default ratio: 16/1
            gradient_checkpointing=True,
            num_train_epochs=20,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            dataloader_drop_last=True,
            push_to_hub=True,
            use_cpu=True,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.datacollator,
            train_dataset=self.training_dataset,
            eval_dataset=self.testing_dataset,
            compute_metrics=self._compute_metrics,
        )
    
    def train(self):
        if self.training_dataset == None or self.testing_dataset == None:
            print("Missing Dataset(s)!")
            return
        self.trainer.train()

# 20250403: table searcher for intent prediction
# Vocab based classifier for exploration and comparison with LLM models, should have the lowest performance (0.525 /w Whipser-small)
class TableSearcher:
    """
    A custom class used to predict the intent from script without using any LLM models

    """

    # Origin table from paper
    # TABLE = {
    #     "ACE Inhibitor": ["高血壓", "血管緊張素轉換酶抑制劑", "高血壓藥", "high blood pressure", "hypertension", "ACE Inhibitor"],
    #     "Metformin": ["糖尿病", "糖尿病藥", "甲福明", "Diabetes", "Metformin"],
    #     "Atorvastatin": ["冠脈病", "冠心病", "心臟病", "Coronary heart disease", "CAD", "阿伐他汀", "心臟病藥", "Atorvastatin"],
    #     "Amitriptyline": ["抑鬱症", "Depression", "Depression disorder", "阿米替林", "抗抑鬱藥", "Amitriptyline", "Antidepressant"]
    # }

    # Optimized table for better performance
    MEDICINE_TABLE = {
        "ACE Inhibitor": ["高血壓", "血管緊張素轉換酶抑制劑", "high blood pressure", "hypertension", "ace inhibitor"],
        "Metformin": ["糖尿病", "甲福明", "diabetes", "metformin"],
        "Atorvastatin": ["冠脈病", "冠心病", "心臟病", "coronary heart disease", "cad", "阿伐他汀", "atorvastatin"],
        "Amitriptyline": ["抑鬱症", "depression", "depression disorder", "阿米替林", "amitriptyline", "antidepressant"]
    }

    # Verb for Intention (Direct & Assertive words): Commands often use imperative verbs (action words) and sound direct.
    VERB_LIST = ["俾", "遞", "交", "攞", "拎", "揸", "執", "抓", "要", "grab", "want", "搵", "need", "find", "get",
            "take", "bring", "fetch", "hand", "give", "pass",
            "offer", "provide", "present", "deliver", 
            "clutch", "grasp", "pick up", "capture" , "where"]

    # Hedging words (Uncertain and Questioning): Confused statements often contain hedging words (maybe, not sure, which one) or question-like structures.
    CONFUSE_LIST = ["邊個", "邊隻", "邊款", "which", "what", "unsure", "uncertain", "not sure", "定係", "不知", "唔知", "or", "either", "maybe", "perhaps",
            "could", "might", "should", "may", "possibly", "likely", "probably", "seem", "seems", "seemingly", "係咪"]

    # Assertive words
    ASSERTIVE_LIST = ["yes", "ok", "係", "好", "yes", "confirm", "affirmative", "assertive", "agree"]

    cache_med_list: list[str]
    is_direct: bool
    is_confused: bool
    is_request_confirm: bool

    def __init__(self):
        self._reset_state()
    
    def _reset_state(self, reset_cache=True):
        """
        Reset the state of the TableSearcher object.
        """
        if not reset_cache:
            self.is_direct = False
            self.is_confused = False
        else:
            self.cache_med_list = []
            self.is_direct = False
            self.is_confused = False
            self.is_request_confirm = False

    def pop_medicine(self) -> str:
        """
        Get the first detected medicine from the cache.
        """
        if self.cache_med_list and self.is_request_confirm == False:
            return self.cache_med_list.pop(0)
        else:
            print("No confirmed medicine detected in the cache.")
        return ""
    
    def pop_medicine_list(self) -> list[str]:
        """
        Get the list of detected medicines from the cache.
        """

        if self.cache_med_list and self.is_request_confirm == False:
            return self.cache_med_list
        else:
            print("No confirmed medicine detected in the cache.")
        return []

    def predict(self, script: str):

        lower_script = script.lower()

        print("Script:", lower_script)
        
        # Medicine Classification: If Medicine name is in the script, append detected_med with the corresponding key
        if not self.is_request_confirm: # Use previous cache if requesting confirm
            self.cache_med_list = TableSearcher._detect_medicine(lower_script)

        # Semantic Analysis: if confused, return; else, see if it is response to confirmation; else, see if requesting a medicine
        self.is_confused = TableSearcher._detect_keyword(lower_script, self.CONFUSE_LIST)
        
        # If the script is a request for confirmation, use the assertive list
        # Otherwise, use the verb list
        if self.is_request_confirm:
                self.is_direct = TableSearcher._detect_keyword(lower_script, self.ASSERTIVE_LIST)
                self.is_request_confirm = False
        else:
            self.is_direct = TableSearcher._detect_keyword(lower_script, self.VERB_LIST)
        return
    
    @staticmethod
    def _detect_keyword(script: str, searching_list: list[str]) -> bool:
        """
        Detects if the script contains any confused words.
        """
        for word in searching_list:
            if re.search(word, script):
                return True
        return False
    
    @staticmethod
    def _detect_medicine(script: str) -> List[str]:
        """
        Detects if the script contains any medicine names.
        """
        detected_med = []
        for key in TableSearcher.MEDICINE_TABLE.keys():
            for word in TableSearcher.MEDICINE_TABLE[key]:
                if re.search(word, script):
                    detected_med.append(key)
                    break
        return detected_med
    
    def generate_response(self) -> int:

        # If detected_med has item(s), and is_direct -> Direct & Valid Command ->                                           "certain"                  
        # If detected_med has item(s), is_confused and is_direct -> soft/polite request -> consider confirmed ->            "certain"
        # If detected_med has item(s), but is_confused -> Don't know if need to take the med / where is it ->               "confused", ask for confirmation
        # If detected_med is empty, but is_direct -> invalid command ->                                                     "invalid command"   
        # Else ->                                                                                                           "invalid input"

        response_type = -1
        has_cache_med = len(self.cache_med_list) > 0
        reset_cache = True

        if has_cache_med and self.is_direct:
            self.is_request_confirm = False
            reset_cache = False
            response_type = 0

        if has_cache_med and self.is_confused and not self.is_direct:
            self.is_request_confirm = True
            reset_cache = False
            response_type = 1

        if not has_cache_med and self.is_direct:
            response_type = 2

        if not has_cache_med and self.is_confused:
            response_type = 3

        self._reset_state(reset_cache=reset_cache)

        return response_type

class Multitask_BERT:

    def __init__(self, repo_id: str, pretrain_model: str, use_exist: bool, dataset: PharmaIntent_Dataset):
        self.model_id = repo_id
        self.dataset = dataset
        self.label_list = dataset.label_list

        # Use existing model or load pre-trained BERT model
        call_id = repo_id if use_exist & (repo_id != None) else pretrain_model

        self._set_processor(pretrain_model)
        self._call_model(call_id)
        self._preprocess_dataset()
        self._prepare_training()

    def _set_processor(self, model_id: str):
        self.tokenizer = BertTokenizer.from_pretrained(model_id)

    def _call_model(self, model_id: str):
        id2label = {i: label for i, label in enumerate(self.label_list)}
        label2id = {label: i for i, label in enumerate(self.label_list)}

        config = BertConfig(
            id2label=id2label,
            label2id=label2id,
            num_labels=len(self.label_list),
            problem_type="multi_label_classification",
            pad_token_id=0,
            classifier_dropout=0.1,
            classifier_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            max_position_embeddings=512,
            vocab_size=len(self.tokenizer),
        )

        self.model = BertForSequenceClassification.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)