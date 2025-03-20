"""


This script is for storing self-defined model classes, NOT FOR CALLING DUE TO CUDA ALLOCATION PROBLEM ATM!


"""
# SET GLOBAL
BATCH_SIZE = 16
EVAL_SIZE = 8
MAX_STEPS = 300
EVAL_STEPS = 30

LLM_EPOCH = 3
ASR_EPOCH = 1

SEED = 42

from typing import List, Optional, Union
import numpy as np
from transformers import (
    WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor,
    BertForSequenceClassification, BertTokenizer, BertConfig,
    Wav2Vec2ForCTC, Wav2Vec2Processor,
    GPT2ForSequenceClassification, GPT2Tokenizer, GPT2Config,
    Trainer, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset, IterableDataset, Audio

from Collators import DataCollatorSpeechSeq2SeqWithPadding, DataCollatorCTCWithPadding
from Datasets import MedIntent_Dataset
import evaluate

# List of testing Whisper models

# Source: https://huggingface.co/openai/whisper-tiny
# Params: 39M (tiny)

# Source: https://huggingface.co/openai/whisper-small
# Params: 244M (small)

# Source: https://huggingface.co/openai/whisper-large-v3
# Params: 1550 M (large-3)

# Source: https://huggingface.co/openai/whisper-large-v3-turbo
# Params: 809 M (large-v3-turbo)

# Custom Repo-id: borisPMC/whisper_grab_medicine_intent

class Whisper_Model:

    model_id: str
    model: WhisperForConditionalGeneration
    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizer

    def __init__(self, repo_id: str, pretrain_model: str, use_exist: bool, dataset: MedIntent_Dataset):
        self.model_id = repo_id
        self.dataset = dataset

        # Use existing model or load pre-trained Whisper model
        call_id = repo_id if use_exist & (repo_id != None) else pretrain_model

        self._import_model_set(call_id)
        self._prepare_datasets()
        self._prepare_training()

    def _import_model_set(self, repo_id):
        self.model = WhisperForConditionalGeneration.from_pretrained(repo_id)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(repo_id)
        self.feature_extractor.chunk_length = 5  # 5 seconds
        self.tokenizer = WhisperTokenizer.from_pretrained(repo_id)

    def _prepare_datasets(self):
        # Cast audio column to 16kHz sampling rate
        self.dataset.train_ds = self.dataset.train_ds.cast_column("Audio", Audio(sampling_rate=16000))
        self.dataset.test_ds = self.dataset.test_ds.cast_column("Audio", Audio(sampling_rate=16000))

        def preprocess_function(batch):
            audio = batch["Audio"]
            label = batch["Speech"]

            # Ensure audio is truncated/padded to 5 seconds (5 * 16000 samples)
            max_length = 5 * 16000  # 5 seconds at 16kHz
            audio_array = [a["array"][:max_length] if len(a["array"]) > max_length else np.pad(
                a["array"], (0, max_length - len(a["array"])), mode="constant") for a in audio]

            # Extract input features from the processed audio
            input_features = self.feature_extractor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features

            # Tokenize labels and pad them to the same length
            labels = self.tokenizer(
                label,
                padding="max_length",
                truncation=True,
                max_length=225,  # Adjust max length as needed
                return_tensors="pt"
            ).input_ids

            # Convert to list for compatibility with datasets
            batch["input_features"] = input_features.tolist()
            batch["labels"] = labels.tolist()
            return batch

        # Apply preprocessing
        self.train_dataset = self.dataset.train_ds.map(preprocess_function, remove_columns=["Audio", "Speech", "Label"], batched=True)
        self.test_dataset = self.dataset.test_ds.map(preprocess_function, remove_columns=["Audio", "Speech", "Label"], batched=True)

    def _prepare_training(self):
        self.cer = evaluate.load("cer")

        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./{self.model_id}",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=EVAL_SIZE,
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

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )

    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # Decode predictions and labels
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        # Compute CER
        cer = self.cer.compute(predictions=pred_str, references=label_str)
        return {"CER": cer}

    def train(self):
        if not self.train_dataset or not self.test_dataset:
            print("Missing Dataset(s)!")
            return
        # self.trainer.train()
        
        # Wrap feature extractor and tokenizer into a prcoessor and push to hub
        processor = WhisperProcessor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        processor.save_pretrained(self.model_id, push_to_hub=True)

class BERT_Model:
    def __init__(self, model_id: str, use_exist: bool, dataset: MedIntent_Dataset):
        self.model_id = model_id
        self.dataset = dataset
        self.label_list = dataset.label_list

        # Use existing model or load pre-trained BERT model
        call_id = model_id if use_exist else "bert-base-multilingual-uncased"

        self._set_processor(call_id)
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

        self.model = BertForSequenceClassification.from_pretrained(model_id, config=config)
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
        self.testing_dataset = self.dataset.test_ds.map(preprocess_function, remove_columns=["Speech", "Label", "Audio"], batched=True)

    def _prepare_training(self):
        self.evaluator = evaluate.load("f1")
        self.data_collator = DataCollatorWithPadding(self.tokenizer, return_tensors="pt")

        training_args = TrainingArguments(
            output_dir=f"./{self.model_id}",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=EVAL_SIZE,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            num_train_epochs=LLM_EPOCH,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            dataloader_drop_last=True,
            push_to_hub=True,
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

class Wav2Vec2_Model:
    def __init__(self, repo_id: str, use_exist: bool, dataset: MedIntent_Dataset):
        self.model_id = repo_id
        self.dataset = dataset

        # Use existing model or load pre-trained Wav2Vec2 model
        call_id = repo_id if use_exist else "facebook/wav2vec2-large-960h"

        self._import_model_set(call_id)
        self._prepare_datasets()
        self._prepare_training()

    def _import_model_set(self, repo_id):
        self.model = Wav2Vec2ForCTC.from_pretrained(repo_id)
        self.processor = Wav2Vec2Processor.from_pretrained(repo_id)

    def _prepare_datasets(self):
        # Cast audio column to 16kHz sampling rate
        self.dataset.train_ds = self.dataset.train_ds.cast_column("Audio", Audio(sampling_rate=16000))
        self.dataset.test_ds = self.dataset.test_ds.cast_column("Audio", Audio(sampling_rate=16000))

        def preprocess_function(batch):
            audio = batch["Audio"]
            label = batch["Speech"]

            # Extract input values from audio
            input_values = self.processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt",
                padding=True
            ).input_values[0]

            # Tokenize labels
            with self.processor.as_target_processor():
                labels = self.processor(label).input_ids

            batch["input_values"] = input_values.tolist()
            batch["labels"] = labels
            return batch

        # Apply preprocessing
        self.train_dataset = self.dataset.train_ds.map(
            preprocess_function, remove_columns=["Audio", "Speech"], batched=True
        )
        self.test_dataset = self.dataset.test_ds.map(
            preprocess_function, remove_columns=["Audio", "Speech"], batched=True
        )

    def _prepare_training(self):
        self.cer_metric = evaluate.load("cer")

        self.data_collator = DataCollatorCTCWithPadding(
            processor=self.processor,
            padding=True,
        )

        training_args = TrainingArguments(
            output_dir=f"./{self.model_id}",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=EVAL_SIZE,
            gradient_accumulation_steps=1,
            fp16=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            num_train_epochs=ASR_EPOCH,
            learning_rate=2e-5,
            weight_decay=0.01,
            save_total_limit=2,
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

    def train(self):
        if not self.train_dataset or not self.test_dataset:
            print("Missing Dataset(s)!")
            return
        self.trainer.train()

        # Save processor
        self.processor.save_pretrained(f"./{self.model_id}")
        self.processor.push_to_hub(repo_id=self.model_id)


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

    def __init__(self, model_id: str, use_exist: bool, dataset_class: MedIntent_Dataset):
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

    def _preprocess_dataset(self, dataset: MedIntent_Dataset):

        # No language tokens for BERT

        train_ds, test_ds = dataset.group_train_test()

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
        vectorized_test_ds = test_ds.map(_f, batched=True, remove_columns=["Label","Speech"] , batch_size=16)

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
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=EVAL_SIZE,
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