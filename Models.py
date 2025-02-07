"""


This script is for storing self-defined model classes, NOT FOR CALLING DUE TO CUDA ALLOCATION PROBLEM ATM!


"""
# SET GLOBAL
BATCH_SIZE = 16
EVAL_SIZE = 8
MAX_STEPS = 300
EVAL_STEPS = 30

from typing import List, Union
import numpy as np
from transformers import (
    WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor,
    BertForSequenceClassification, BertTokenizer, BertConfig,
    Wav2Vec2ForCTC, Wav2Vec2Processor,
    GPT2ForSequenceClassification, GPT2Tokenizer, GPT2Config,
    Trainer, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset, IterableDataset, Audio

from Collators import DataCollatorSpeechSeq2SeqWithPadding, DataCollatorCTCWithPadding
from Datasets import CV17_dataset, Custom_Dataset
import evaluate

# Source: https://huggingface.co/openai/whisper-small
# Params: 242M (small)

# Custom Repo-id: borisPMC/whisper_grab_medicine_intent

class Whisper_Model:

    # Pre-defined for clean structure
    model_id: str
    model: WhisperForConditionalGeneration
    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizer

    train_dataset: Union[Dataset, IterableDataset]
    test_dataset: Union[Dataset, IterableDataset]

    data_collator: DataCollatorSpeechSeq2SeqWithPadding
    trainer: Seq2SeqTrainer

    def __init__(self, repo_id: str, use_exist: bool, ds: CV17_dataset):

        self.model_id = repo_id
        if use_exist:
            call_id = repo_id
        else:
            call_id = "openai/whisper-small" # default pre-train model
        
        self._import_ds(ds)
        self._import_model_set(call_id)
        self._prepare_training()

    def _import_ds(self, ds: CV17_dataset):

        ds_train_casted = ds.train_ds.cast_column("audio", Audio(sampling_rate=16000))
        ds_test_casted = ds.test_ds.cast_column("audio", Audio(sampling_rate=16000))

        def _f(batch: dict):
            
            audio_col = batch["audio"]
            sentence_col = batch["sentence"]

            input_features_col = []
            labels_col = []
            
            for audio in audio_col:
                features = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
                input_features_col.append(features)

            for sentence in sentence_col:
                label = self.tokenizer(sentence).input_ids
                labels_col.append(label)

            batch["input_features"] = input_features_col
            batch["labels"] = labels_col

            return batch
        
        # Input cols: audio, sentence
        ds_train_vectorized = ds_train_casted.map(_f,batched=True, remove_columns=["audio", "sentence"], batch_size=BATCH_SIZE)
        ds_test_vectorized = ds_test_casted.map(_f, batched=True, remove_columns=["audio", "sentence"], batch_size=BATCH_SIZE)

        self.train_dataset = ds_train_vectorized
        self.test_dataset = ds_test_vectorized

    # Initalisation on Model, FEx, and Tkn
    def _import_model_set(self, repo_id):
        self.model = WhisperForConditionalGeneration.from_pretrained(repo_id)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(repo_id)
        self.tokenizer = WhisperTokenizer.from_pretrained(repo_id)

    # Initialisation on DataCollator, EvaluationModule and Trainer
    def _prepare_training(self):
        
        self.cer = evaluate.load("cer")

        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir= ("./" + self.model_id),  # change to a repo name of your choice
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=EVAL_SIZE,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size Default ratio: 16/1
            gradient_checkpointing=True,
            fp16=True,
            eval_strategy="steps",
            predict_with_generate=True,
            generation_max_length=225,
            max_steps=MAX_STEPS, #500*16=8k rows stepped
            save_steps=EVAL_STEPS,
            eval_steps=EVAL_STEPS,
            # log_level="error", # silence warning spam
            logging_steps=EVAL_STEPS,
            greater_is_better=False,
            push_to_hub=True,
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.feature_extractor,
        )

        self.trainer = trainer

    # Special name: overwrite Trainer's compute_metrics
    def compute_metrics(self, pred):

        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # Decode the predictions and labels
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        # Compute CER
        cer = self.cer.compute(predictions=pred_str, references=label_str)

        return {"CER: ": cer}
    
    def train(self):
        
        if self.train_dataset == None or self.test_dataset == None:
            print("Missing Dataset(s)!")
            return
        
        self.trainer.train()

class BERT_Model():

    model_id: str
    model: BertForSequenceClassification
    tokenizer: BertTokenizer

    training_dataset: Union[Dataset, IterableDataset]
    testing_dataset: Union[Dataset, IterableDataset]

    datacollator: DataCollatorWithPadding
    evaluator: evaluate.EvaluationModule
    trainer: Trainer

    label_list: List[str]

    def __init__(self, model_id: str, use_exist: bool, dataset_class: Custom_Dataset):
        self.model_id = model_id
        self.label_list = dataset_class.class_list

        if use_exist:
            call_id = self.model_id
        else:
            call_id = "bert-base-multilingual-uncased"

        self._set_processor(call_id)
        self._call_model(call_id)
        self._preprocess_dataset(dataset_class)

        # Define first avoid CUDA error
        self._prepare_training()

    def _set_processor(self, model_id: str):
        self.tokenizer = BertTokenizer.from_pretrained(model_id)

    def _call_model(self, model_id: str):

        # Set labels for the model

        id2label = {}
        label2id = {}

        for i in range(len(self.label_list)):
            id2label[i] = self.label_list[i]
            label2id[self.label_list[i]] = i
        
        config = BertConfig(
            id2label = id2label,
            label2id = label2id,
            num_labels = 5
        )
        
        self.model = BertForSequenceClassification.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)
        self._resize_token_embeddings()

    # Ensure the model's token embeddings match the tokenizer's vocabulary size (due default: 2 classes | this ds: 5 classes)
    def _resize_token_embeddings(self):
        if self.tokenizer and self.model:
            vocab_size = len(self.tokenizer)
            self.model.resize_token_embeddings(vocab_size)

    def _preprocess_dataset(self, dataset: Custom_Dataset):

        # No language tokens for BERT

        train_ds, test_ds = dataset.group_train_test()

        def _f(batch):

            speech_col = batch["Speech"]
            label_col = batch["Label"]

            tokenised_dict = self.tokenizer(speech_col, padding="max_length", return_tensors="pt")
            # output_ids = self.tokenizer(label_col, padding="max_length", return_tensors="pt").input_ids

            batch["input_ids"] = tokenised_dict.input_ids
            batch["token_type_ids"] = tokenised_dict.token_type_ids
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

class Wav2Vec2_Model:
    def __init__(self, repo_id: str, use_exist: bool, ds: CV17_dataset):
        self.model_id = repo_id
        self.ds_class = ds

        self._import_model_set(repo_id)

        self.data_collator = DataCollatorCTCWithPadding(
            processor=self.processor,
            padding=True,
        )

        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

        self._prepare_training()

    def _import_model_set(self, repo_id):
        self.model = Wav2Vec2ForCTC.from_pretrained(repo_id)
        self.processor = Wav2Vec2Processor.from_pretrained(repo_id)

    def _prepare_training(self):
        training_args = TrainingArguments(
            output_dir=("./" + self.model_id),
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=1,
            fp16=True,
            evaluation_strategy="steps",
            save_steps=EVAL_STEPS,
            eval_steps=EVAL_STEPS,
            logging_steps=EVAL_STEPS,
            learning_rate=2e-5,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=2,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.ds_class.train_ds,
            eval_dataset=self.ds_class.test_ds,
            data_collator=self.data_collator,
            tokenizer=self.processor.feature_extractor,
            compute_metrics=self.compute_metrics
        )

    def preprocess_function(self, batch):
        input_values_col = []
        labels_col = []

        for audio in batch["audio"]:
            input_values = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
            input_values_col.append(input_values)

        for sentence in batch["sentence"]:
            with self.processor.as_target_processor():
                labels = self.processor(sentence).input_ids
            labels_col.append(labels)

        batch["input_values"] = input_values_col
        batch["labels"] = labels_col

        return batch

    def prepare_datasets(self):
        for key in self.ds_class.lang_list:
            self.ds_class.ds_by_lang[key]["train"] = self.ds_class.ds_by_lang[key]["train"].cast_column("audio", Audio(sampling_rate=16000))
            self.ds_class.ds_by_lang[key]["eval"] = self.ds_class.ds_by_lang[key]["eval"].cast_column("audio", Audio(sampling_rate=16000))

            self.ds_class.ds_by_lang[key]["train"] = self.ds_class.ds_by_lang[key]["train"].map(
                self.preprocess_function, batched=True, remove_columns=["audio", "sentence"], batch_size=16
            )
            self.ds_class.ds_by_lang[key]["eval"] = self.ds_class.ds_by_lang[key]["eval"].map(
                self.preprocess_function, batched=True, remove_columns=["audio", "sentence"], batch_size=16
            )

    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions.argmax(-1)

        pred_str = self.processor.batch_decode(pred_ids)
        labels_ids[labels_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(labels_ids, group_tokens=False)

        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer, "wer": wer}

    def train(self):
        if self.ds_class.train_ds is None or self.ds_class.test_ds is None:
            print("Missing Dataset(s)!")
            return
        self.trainer.train()


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

    def __init__(self, model_id: str, use_exist: bool, dataset_class: Custom_Dataset):
        self.model_id = model_id
        self.label_list = dataset_class.class_list

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

    def _preprocess_dataset(self, dataset: Custom_Dataset):

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