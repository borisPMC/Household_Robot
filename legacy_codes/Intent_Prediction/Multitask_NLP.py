
"""
Multitask model for two tasks: intent classification and named entity recognition.
This model uses BERT (or alternatives) as the base model and adds two separate heads for each task.

!!! LEGACY CODE !!!


"""
import asyncio
from dataclasses import dataclass
import json
import re
from typing import Any, Dict, List
import evaluate
import numpy as np
from transformers import (
    BertTokenizer, BertConfig, BertModel, Trainer, TrainingArguments, 
    EvalPrediction, AutoConfig, AutoModel, Pipeline, PreTrainedModel,
    PretrainedConfig, BertForSequenceClassification, BertForTokenClassification)
import torch
from torch import nn
from datasets import DatasetDict
from Datasets import IntentDataset, call_dataset

"""
CLASS LABEL LIST
"""

# INTENT_LABEL = ["other_intents", "retrieve_med", "search_med", "enquire_suitable_med"]
INTENT_LABEL = [
    "call_contact",
    "retrieve_med",
    "enquire_location",
    "enquire_suitable_med",
    "enquire_time",
    "enquire_weather",
    "general_chat",
    "enquire_info",
    "retrieve_other",
    "set_furniture",
    "set_software",
]

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
        # self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intent_labels)
        self.ner_classifier_1 = nn.Linear(self.bert.config.hidden_size, num_ner_labels)
        self.ner_classifier_2 = nn.Linear(self.bert.config.hidden_size, num_ner_labels)

        self.intent_softmax = nn.Softmax(dim=1)
        self.ner_softmax = nn.Softmax(dim=1)


    def forward(self, input_ids, attention_mask, intent_labels=None, ner_labels=None):
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

        sequence_output = outputs.last_hidden_state  # For NER, token level
        ner_logits_1 = self.ner_classifier_1(sequence_output)
        ner_logits_1 = self.ner_softmax(ner_logits_1)
        ner_logits_2 = self.ner_classifier_2(sequence_output)
        ner_logits_2 = self.ner_softmax(ner_logits_2)

        ner_logits = ner_logits_1 + ner_logits_2
        ner_probs = self.ner_softmax(ner_logits)

        pooled_output = outputs.pooler_output       # For intent classification, sequence level
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        intent_probs = self.intent_softmax(intent_logits) # total prob = 1 for each token

        loss = None
        if ner_labels is not None and intent_labels is not None:
            loss_fct_cls = nn.CrossEntropyLoss()
            loss_fct_tok = nn.CrossEntropyLoss(ignore_index=-100)
            loss_intent = loss_fct_cls(intent_logits, intent_labels)
            loss_token = loss_fct_tok(ner_logits_2.view(-1, ner_logits_2.shape[-1]), ner_labels.view(-1))
            loss = loss_intent + loss_token

        # Prepare output dictionary (Use softmax -> probs instead)
        output = {
            "intent_probs": intent_probs,
            "ner_probs": ner_probs,
            "loss": loss,
        }

        return output

class Multitask_BERT_v2(PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
            
        super().__init__(PretrainedConfig())
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models. 

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name, 
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)

@dataclass
class DataCollatorForMultiTaskBert:

    def __init__(self, tokenizer: BertTokenizer, label_pad_token_id: int = -100, max_length = 128):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate features by task type

        # raise Exception(f"Error: {features}")

        intent_labels = [f.pop("intent_labels") for f in features]
        ner_labels = [f.pop("ner_labels") for f in features]

        # Pad inputs
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )

        # Pad token labels
        max_len = self.max_length
        padded_ner_labels = [(tl + [self.label_pad_token_id] * (max_len - len(tl))) for tl in ner_labels]
        
        # raise Exception(f"Check {ner_labels}, {padded_ner_labels}")
        batch["input_ids"] = batch["input_ids"].squeeze(1)
        batch["attention_mask"] = batch["attention_mask"].squeeze(1)
        batch["intent_labels"] = torch.tensor(intent_labels, dtype=torch.long)
        batch["ner_labels"] = torch.tensor(padded_ner_labels, dtype=torch.long)

        return batch
    


"""
Preprocessing functions for the dataset.
"""

def hybrid_split(string: str) -> List[str]:

    regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
    matches = re.findall(regex, string, re.UNICODE)
    return matches

def set_trainer(model, tokenizer, evaluators, train_ds, eval_ds, max_length=128) -> Trainer:

    def compute_metrics(pred: EvalPrediction):
        """
        Compute metrics for multitask learning. *Evaluation on Token-level, not speech level*

        Args:
            pred: A tuple containing predictions and labels:
                - probs: A tuple of (intent_probs, ner_probs).
                - labels: A tuple of (intent_labels, ner_labels).

        Returns:
            dict: A dictionary containing accuracy metrics for intent and NER.
        """

        intent_preds = pred.predictions[0]
        ner_preds = pred.predictions[1]

        intent_labels = pred.label_ids[0]
        ner_labels = pred.label_ids[1]

        # raise Exception (f"{intent_preds} | {ner_preds} | {intent_labels} | {ner_labels}")

        # ----- Intent Classification -----
        intent_preds = np.argmax(intent_preds, axis=1)
        intent_acc = evaluators["f1_metric"].compute(predictions=intent_preds, references=intent_labels, average="macro")

        # ----- Token Classification -----
        ner_preds = np.argmax(ner_preds, axis=2)

        # Convert token predictions and labels to string for seqeval
        true_token_labels = []
        pred_token_labels = []

        true_med_request = []
        pred_med_request = []

        # raise Exception(ner_preds, ner_labels)

        for pred_seq, label_seq in zip(ner_preds, ner_labels):
            true_seq = []
            pred_seq_str = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue  # ignore padding tokens
                true_seq.append(str(l))
                pred_seq_str.append(str(p))
            true_token_labels.append(true_seq)
            pred_token_labels.append(pred_seq_str)
            # Convert the medical request
            true_med_request.append(IntentDataset.check_NER(true_seq))
            pred_med_request.append(IntentDataset.check_NER(pred_seq_str))

        # raise Exception(pred_token_labels, true_token_labels)

        # raise Exception(f"{true_med_request}, {pred_med_request}")

        ner_seq_scores = evaluators["seq_f1_metric"].compute(predictions=pred_med_request, references=true_med_request)

        flatten_true_token_labels = [item for sub in true_token_labels for item in sub]
        flatten_pred_token_labels = [item for sub in pred_token_labels for item in sub]

        ner_tok_scores = evaluators["f1_metric"].compute(predictions=flatten_pred_token_labels, references=flatten_true_token_labels, average="macro")

        return {
            "intent_f1": intent_acc["f1"],
            "ner_tok_f1": ner_tok_scores["f1"],
            "ner_seq_f1": ner_seq_scores["overall_f1"],
        }

    training_args = TrainingArguments(
        output_dir="temp/multitask_BERT_MedicGrabber",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
        push_to_hub=True,
        hub_model_id="borisPMC/multitask_BERT_MedicGrabber",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForMultiTaskBert(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer

### LEGACY CODE ###
async def main():

    model_name  = "bert-base-uncased" # "bert-base-uncased" # "bert-base-multilingual-cased"

    tokenizer = BertTokenizer.from_pretrained(model_name)

    ds = call_dataset()

    evaluators = {
        "f1_metric": evaluate.load("f1"),
        "seq_f1_metric": evaluate.load("seqeval")
    }

    model = Multitask_BERT(
        model_name=model_name,
        num_intent_labels=len(INTENT_LABEL),
        num_ner_labels=len(NER_LABEL),
    )

    def preprocess(example, tokenizer, max_length=128):

        text = example["Text"]

        # Tokenize the text
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # Prepare the item

        item = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "intent_labels": example["Intent_Label"],
            "ner_labels": example["NER_Labels"]
        }

        # raise Exception(f"Exception: {item}")

        return item

    trainer = None

    for lang in ds.datasets.keys():
        # Train the model on each dataset
        ds.set_splits_by_lang(lang)
        feed_train = ds.train_ds.map(preprocess, fn_kwargs={"tokenizer": tokenizer})
        feed_valid = ds.valid_ds.map(preprocess, fn_kwargs={"tokenizer": tokenizer})
        if trainer == None:
            trainer = set_trainer(model, tokenizer, evaluators, feed_train, feed_valid)
            trainer.train(resume_from_checkpoint=False)
        else:
            trainer.train_dataset = feed_train
            trainer.eval_dataset = feed_valid
            trainer.train(resume_from_checkpoint=True)
    
    trainer.push_to_hub()
    # tokenizer.push_to_hub("borisPMC/multitask_BERT_MedicGrabber_2")

    # generate_multitask_bert_config(config, "multitask_BERT_MedicGrabber", len(IntentDataset.INTENT_LABEL), len(IntentDataset.NER_LABEL))
    
    # Upload final model
    # hf_model = BertModel.from_pretrained("./temp/multitask_BERT_MedicGrabber/checkpoint-170")

    # hf_model.push_to_hub("borisPMC/multitask_BERT_MedicGrabber", commit_message="Uploading Multitask BERT model")
    # tokenizer.push_to_hub("borisPMC/multitask_BERT_MedicGrabber")

if __name__ == "__main__":
    asyncio.run(main())