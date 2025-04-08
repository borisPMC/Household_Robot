
"""
Multitask model for two tasks: intent classification and named entity recognition.
This model uses BERT (or alternatives) as the base model and adds two separate heads for each task.

"""

import re
from typing import List
import evaluate
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertConfig, BertModel, Trainer, TrainingArguments, EvalPrediction
import torch
from torch import nn
from torch.utils.data import Dataset
from datasets import DatasetDict
from sklearn.model_selection import train_test_split
from Datasets import New_PharmaIntent_Dataset, call_dataset

"""
CLASS LABEL LIST
"""

INTENT_LABEL = ["other_intents", "retrieve_med", "search_med", "enquire_suitable_med"]

# O: irelevant B: beginning I: inside
NER_LABEL = ["O", "B-ACE_Inhibitor", "I-ACE_Inhibitor", "B-Metformin", "I-Metformin", "B-Atorvastatin", "I-Atorvastatin", "B-Amitriptyline", "I-Amitriptyline",]




class Multitask_BERT(nn.Module):
    def __init__(self, model_name: str, num_intent_labels: int, num_ner_labels: int):
        """
        Multitask BERT model with two heads: one for intent classification and one for NER.

        Args:
            model_name (str): Pretrained BERT model name (e.g., "bert-base-uncased").
            num_intent_labels (int): Number of intent classification labels.
            num_ner_labels (int): Number of NER labels.
        """
        super(Multitask_BERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intent_labels)
        self.ner_classifier = nn.Linear(self.bert.config.hidden_size, num_ner_labels)
        
        self.intent_softmax = nn.Softmax(dim=1)
        self.ner_softmax = nn.Softmax(dim=1)


    def forward(self, input_ids, attention_mask, intent_label=None, ner_labels=None):
        """
        Forward pass for the multitask model.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            intent_labels (torch.Tensor, optional): Ground truth intent labels for loss computation.
            ner_labels (torch.Tensor, optional): Ground truth NER labels for loss computation.

        Returns:
            dict: A dictionary containing logits and optionally losses for both tasks.
        """
        # Pass inputs through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state  # For NER
        ner_logits = self.ner_classifier(sequence_output)
        ner_probs = self.ner_softmax(ner_logits)

        pooled_output = outputs.pooler_output       # For intent classification
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        intent_probs = self.intent_softmax(intent_logits)

        # Prepare output dictionary
        output = {
            "intent_logits": intent_logits,
            "ner_logits": ner_logits,
            "intent_probs": intent_probs,
            "ner_probs": ner_probs
        }

        # Compute losses if labels are provided
        if intent_label is not None and ner_labels is not None:
            intent_loss_fn = nn.CrossEntropyLoss()
            ner_loss_fn = nn.CrossEntropyLoss()

            intent_loss = intent_loss_fn(intent_logits, intent_label)
            # Flatten NER logits and labels for loss computation
            ner_loss = ner_loss_fn(ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1))

            output["loss"] = intent_loss + ner_loss # Total loss

        return output

class MultitaskDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int, config: dict):
        """
        Dataset for multitask learning with intent classification and NER.

        Args:
            file_path (str): Path to the dataset file (e.g., multitask_ds.xlsx).
            tokenizer (BertTokenizer): Pretrained BERT tokenizer.
            max_length (int): Maximum sequence length for tokenization.
            config (dict): Configuration dictionary with keys:
                - "language": Language to filter the dataset (e.g., "English", "Cantonese").
                - "train_ratio": Ratio for training data split (e.g., 0.8).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load and preprocess the dataset
        self.data = pd.read_excel(
            file_path, 
            dtype=str,
            usecols=["Speech", "Intent", "NER_Tag", "Major_Language", "Audio_Path"],
            skiprows=[lambda x: x["Major_Language"] != config["language"]])
        
        # Split the dataset into train, validation, and test sets
        train_ratio = config.get("train_ratio", 0.8)
        train_data, test_valid_data = train_test_split(self.data, test_size=1 - train_ratio, random_state=42, shuffle=True)
        valid_data, test_data = train_test_split(test_valid_data, test_size=0.5, random_state=42, shuffle=True)

        self.train_data = train_data.reset_index(drop=True)
        self.valid_data = valid_data.reset_index(drop=True)
        self.test_data = test_data.reset_index(drop=True)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:

        tokenized_speech = []
        ner_labels = []

        for _, row in df.iterrows():
            speech = row["Speech"]
            ner_tag = row["NER_Tag"]

            # Tokenize the speech
            tokens = hybrid_split(speech)
            tokenized_speech.append(tokens)

            # Ensure NER_Tag length matches the number of tokens
            if len(ner_tag) != len(tokens):
                raise ValueError(f"Mismatch between tokens and NER_Tag: {speech}")

            ner_labels.append([int(tag) for tag in ner_tag])

        df["Tokenized_Speech"] = tokenized_speech
        df["NER_Labels"] = ner_labels
        return df

    def get_split(self, split: str) -> Dataset:

        if split == "train":
            return self._prepare_data(self.train_data)
        elif split == "valid":
            return self._prepare_data(self.valid_data)
        elif split == "test":
            return self._prepare_data(self.test_data)
        else:
            raise ValueError("Invalid split name. Choose from 'train', 'valid', or 'test'.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single data point for training.

        Args:
            idx (int): Index of the data point.

        Returns:
            dict: A dictionary containing input IDs, attention mask, intent label, and NER labels.
        """
        row = self.data.iloc[idx]
        text = row["Speech"]
        intent_label = INTENT_LABEL.index(row["Intent"])
        ner_labels = [int(tag) for tag in row["NER_Tag"]]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Prepare the item
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "intent_label": torch.tensor(intent_label, dtype=torch.long),
            "ner_labels": torch.tensor(ner_labels, dtype=torch.long),
        }
        return item
    
"""
Preprocessing functions for the dataset.
"""

def hybrid_split(string: str) -> List[str]:

    regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
    matches = re.findall(regex, string, re.UNICODE)
    return matches

def train_multitask_model(model, tokenizer, evaluator, dataset: DatasetDict, max_length=128):

    def align_predictions_by_row(predictions, labels):
        """
        Align predictions and labels to the original rows using tokenizer's word_ids.

        Args:
            predictions (np.ndarray): Model's token-level predictions.
            labels (np.ndarray): Token-level ground truth labels.

        Returns:
            tuple: Aligned predictions and labels at the row level.
        """
        aligned_preds = []
        aligned_labels = []

        for pred, label in zip(predictions, labels):
            word_ids = tokenizer.word_ids()  # Get word IDs for the current input
            row_preds = []
            row_labels = []

            previous_word_id = None
            current_word_preds = []
            current_word_labels = []

            for token_pred, token_label, word_id in zip(pred, label, word_ids):
                if word_id is None:
                    continue  # Skip special tokens like [CLS] and [SEP]

                if word_id != previous_word_id:
                    # Start of a new word
                    if current_word_preds:
                        # Append the aggregated prediction and label for the previous word
                        row_preds.append(current_word_preds)
                        row_labels.append(current_word_labels)
                    current_word_preds = []
                    current_word_labels = []

                current_word_preds.append(token_pred)
                current_word_labels.append(token_label)
                previous_word_id = word_id

            # Append the last word's predictions and labels
            if current_word_preds:
                row_preds.append(current_word_preds)
                row_labels.append(current_word_labels)

            aligned_preds.append(row_preds)
            aligned_labels.append(row_labels)

        return aligned_preds, aligned_labels

    def compute_metrics(pred: EvalPrediction):
        """
        Compute metrics for multitask learning.

        Args:
            pred: A tuple containing predictions and labels:
                - logits: A tuple of (intent_logits, ner_logits).
                - labels: A tuple of (intent_labels, ner_labels).

        Returns:
            dict: A dictionary containing accuracy metrics for intent and NER.
        """
        # Unpack predictions and labels per batch (16)
        logits, labels = pred.predictions, pred.label_ids 
        
        # logits:   (79*4, 79*128*9, 5, 5)
        # labels:   (79, 79*128)

        if isinstance(logits, tuple) and len(logits) == 4:
            intent_probs, ner_probs = logits[2], logits[3]
        else:
            raise ValueError(f"Unexpected logics structure: {len(logits[1][0][0])}")
        
        intent_labels, ner_labels = labels[0], labels[1]
        intent_preds = np.argmax(intent_probs, axis=1)
        ner_pred = np.argmax(ner_probs, axis=2)  # Get predicted NER labels

        # Intent classification metrics
        intent_accuracy = evaluator.compute(predictions=intent_preds, references=intent_labels, average="weighted") 

        # Flatten predictions and labels for token-level comparison
        ner_preds_flat = ner_pred.flatten()
        ner_labels_flat = ner_labels.flatten()

        # ner_preds_detected_med = New_PharmaIntent_Dataset.check_NER("".join(map(str, ner_preds_flat)))
        # ner_label_detected_med = New_PharmaIntent_Dataset.check_NER("".join(map(str, ner_labels_flat)))
        ner_accuracy = evaluator.compute(predictions=ner_preds_flat, references=ner_labels_flat, average="weighted")
        # print(ner_preds_detected_med, ner_label_detected_med)

        return {
            "intent_accuracy": intent_accuracy,
            "ner_accuracy": ner_accuracy
        }
    
    def preprocess(example, tokenizer, max_length):

        text = example["Speech"]
        intent_label = INTENT_LABEL.index(example["Intent"])

        if type(example["NER_Labels"][0]) == list:
            ner_labels = example["NER_Labels"][0] + ([0] * (max_length - len(example["NER_Labels"][0]))) # padding
        elif type(example["NER_Labels"][0]) == int:
            ner_labels = example["NER_Labels"] + ([0] * (max_length - len(example["NER_Labels"]))) # padding
        else:
            raise Exception("Preprocess fail")

        # Tokenize the text
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        ner_labels = [torch.tensor(tag, dtype=torch.short) for tag in ner_labels]


        # Prepare the item
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "intent_label": torch.tensor(intent_label, dtype=torch.short),
            "ner_labels": ner_labels,
        }
        return item

    processed_ds = dataset.map(preprocess, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length})

    training_args = TrainingArguments(
        output_dir="Intent_Prediction/results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        push_to_hub=True,
        hub_model_id="multitask_BERT_MedicGrabber",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_ds["train"],
        eval_dataset=processed_ds["valid"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def main():
    
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    ds = call_dataset()

    evaluator = evaluate.load("f1")

    model = Multitask_BERT(
        model_name=model_name,
        num_intent_labels=len(INTENT_LABEL),  # Replace with the actual number of intent labels
        num_ner_labels=len(NER_LABEL),   # Replace with the actual number of NER labels
    )

    for lang in ds.datasets.keys():
        # Train the model on each dataset
        lang_ds = ds.datasets[lang].map(New_PharmaIntent_Dataset.postdownload_process)
        train_multitask_model(model, tokenizer, evaluator, lang_ds)

if __name__ == "__main__":
    main()