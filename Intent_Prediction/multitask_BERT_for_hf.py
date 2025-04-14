from dataclasses import dataclass
import inspect
import json
import re
from typing import Any, Dict, List
import evaluate
import numpy as np
from transformers import (
    BertTokenizer, BertConfig, BertModel, Trainer, TrainingArguments, BertTokenizerFast,
    EvalPrediction, AutoConfig, AutoModel, Pipeline, PreTrainedModel,
    PretrainedConfig, BertForSequenceClassification, BertForTokenClassification)
import torch
from torch import nn
from datasets import DatasetDict
from Datasets import New_PharmaIntent_Dataset, call_dataset
import dataclasses
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollator, InputDataClass, DefaultDataCollator, DataCollatorWithPadding
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict

"""
CLASS LABEL LIST
"""

INTENT_LABEL = ["other_intents", "retrieve_med", "search_med", "enquire_suitable_med"]

# O: irelevant B: beginning I: inside
NER_LABEL = ["O", "B-ACE_Inhibitor", "I-ACE_Inhibitor", "B-Metformin", "I-Metformin", "B-Atorvastatin", "I-Atorvastatin", "B-Amitriptyline", "I-Amitriptyline",]

class Multitask_BERT_v2(PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
            
        super().__init__(BertConfig())
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

    def forward(self, task_name, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get the task-specific model
        task_model = self.taskmodels_dict[task_name]
        
        # Filter out unexpected arguments
        model_forward_args = inspect.signature(task_model.forward).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in model_forward_args}

        # Forward pass through the task-specific model
        outputs = self.taskmodels_dict[task_name](
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **filtered_kwargs,
        )

        # If labels are provided, the task-specific model should return a loss
        loss = outputs.loss if labels is not None else None
        logits = outputs.logits

        # Return a dictionary with loss and logits
        return {"loss": loss, "logits": logits}
    
class GrabberBertDataCollator(DefaultDataCollator):
    """
    Extending the existing DataCollator to work with NLP dataset batches
    """
    def collate_batch(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
        first = features[0]
        batch = {}

        if isinstance(first, dict):
            if "labels" in first and first["labels"] is not None:
                # For Intent
                if isinstance(first["labels"], int):
                        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
                # For NER
                elif isinstance(first["labels"], torch.Tensor):
                    labels = torch.stack([f["labels"] for f in features])
                else:
                    raise ValueError("Labels must be a int or tensor")
                batch = {"labels": labels}
            for k, v in first.items():
                if k != "labels" and v is not None and not isinstance(v, str):
                    batch[k] = torch.stack([f[k] for f in features])
            return batch
        else:
            # otherwise, revert to using the default collate_batch
            return DefaultDataCollator().collate_batch(features)


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """
    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """
    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)
    
    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader) 
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) 
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader) 
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])    

class MultitaskTrainer(Trainer):

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        else:
            # Disable distributed training for local runs
            train_sampler = RandomSampler(train_dataset)

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator.collate_batch,
            ),
        )

        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each 
        task Dataloader
        """
        return MultitaskDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset)
            for task_name, task_dataset in self.train_dataset.items()
        })
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom compute_loss method to handle multitask models.
        """
        # Extract the task_name from the inputs
        if "task_name" not in inputs.keys():
            if inputs["labels"].size(dim=0) == len(INTENT_LABEL):
                inputs["task_name"] = "intent"
            else:
                inputs["task_name"] = "ner"

        # Pass the task_name to the model's forward method
        try:
            outputs = model(**inputs)
        except ValueError as e:
            inputs["task_name"] = "intent" if inputs["task_name"] == "ner" else "ner"
            outputs = model(**inputs)

        # Compute the loss
        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss

def prepare_datasets(lang_ds: New_PharmaIntent_Dataset, tokenizer: BertTokenizer, max_length: int):
    """
    Prepares train and evaluation datasets for a given language.

    Args:
        language (str): The language to filter the dataset (e.g., "Cantonese").
        tokenizer (BertTokenizerFast): The tokenizer to preprocess the dataset.
        max_length (int): The maximum sequence length for tokenization.

    Returns:
        tuple: A tuple containing:
            - train_dataset (dict): The training dataset for each task.
            - eval_dataset (dict): The evaluation dataset for each task.
    """
    # Load the dataset
    ds = lang_ds
    ds_intent = DatasetDict({
        "train": ds.train_ds,
        "valid": ds.valid_ds,
        "test": ds.test_ds,
    })
    ds_ner = DatasetDict({
        "train": ds.train_ds,
        "valid": ds.valid_ds,
        "test": ds.test_ds,
    })

    def convert_to_intent_features(batch):
        speech = batch["Speech"]
        intent_labels = batch["Intent_Label"]
        outputs = tokenizer(
            speech, max_length=max_length, padding="max_length", truncation=True
        )
        features = {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
            "labels": intent_labels,
        }
        return features

    def convert_to_ner_features(batch):
        speech = batch["Speech"]
        ner_labels = batch["NER_Labels"]
        outputs = tokenizer(
            speech, max_length=max_length, padding="max_length"
        )
        labels = []
        try:
            for i, label in enumerate(ner_labels):
                word_ids = outputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx and word_idx < len(label):  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Speech: {speech}")
            print(f"NER Labels: {ner_labels}")
            print(f"word_idx: {word_idx}")
            print(f"label: {label}")
            print(f"label_ids: {label_ids}")
            print(f"labels: {labels}")
            raise

        features = {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
            "labels": labels,
        }
        return features

    convert_func_dict = {
        "intent": convert_to_intent_features,
        "ner": convert_to_ner_features,
    }
    columns_dict = {
        "intent": ['input_ids', 'attention_mask', 'labels'],
        "ner": ['input_ids', 'attention_mask', 'labels'],
    }

    # Preprocess the dataset
    dataset_dict = {
        "intent": ds_intent,
        "ner": ds_ner,
    }

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
            )
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=columns_dict[task_name],
            )

    train_dataset = {
        task_name: dataset["train"]
        for task_name, dataset in features_dict.items()
    }

    eval_dataset = {
        task_name: dataset["valid"]
        for task_name, dataset in features_dict.items()
    }

    return train_dataset, eval_dataset

def train_mtmodel(model, tokenizer, evaluators, train_ds, valid_ds):

    train_dataset, eval_dataset = train_ds, valid_ds
    
    def compute_metrics(eval_pred: EvalPrediction):

        preds, labels = eval_pred.predictions, eval_pred.label_ids

        intent_dict = {"preds": [], "labels": [],}
        ner_dict = {"preds": [], "labels": [],}
        outputs = {}

        # Classify the predictions and labels into intent and ner, and align them for evaluation
        for pred, label in zip(preds, labels):
            try:
                if len(pred) == 4:
                    intent_dict["preds"].append(pred.argmax(axis=0))
                    intent_dict["labels"].append(label)
                else:
                    ner_dict["preds"].append(pred.argmax(axis=1))
                    ner_dict["labels"].append(label)
                # else:
                #     raise Exception(f"{pred} | {label}")
            except Exception as e:
                print(f"Error: {e}")
                print(f"Preds: {preds}, Labels: {labels}")
                print(f"Intent Dict: {intent_dict}, NER Dict: {ner_dict}")
                print(f"Pred: {pred}, Label: {label}")
        
        if len(intent_dict["preds"]) > 0:
            # Intent classification
            I_f1 = evaluators["f1_metric"].compute(
                predictions=intent_dict["preds"], 
                references=intent_dict["labels"], 
                average="macro",
            )
            outputs["f1"] = I_f1["f1"]
        else:
            # NER classification
            # Convert token predictions and labels to string for seqeval
            true_token_labels = []
            pred_token_labels = []

            true_med_request = []
            pred_med_request = []

            # raise Exception(ner_preds, ner_labels)
            for pred_seq, label_seq in zip(ner_dict["preds"], ner_dict["labels"]):
                # raise Exception(f"{pred_seq} | {label_seq}")
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
                true_med_request.append(New_PharmaIntent_Dataset.check_NER(true_seq))
                pred_med_request.append(New_PharmaIntent_Dataset.check_NER(pred_seq_str))

            ner_seq_scores = evaluators["seqeval"].compute(predictions=pred_med_request, references=true_med_request)

            flatten_true_token_labels = [item for sub in true_token_labels for item in sub]
            flatten_pred_token_labels = [item for sub in pred_token_labels for item in sub]

            ner_tok_scores = evaluators["f1_metric"].compute(predictions=flatten_pred_token_labels, references=flatten_true_token_labels, average="macro")
            outputs["tkn_f1"] = ner_tok_scores["f1"]
            outputs["med_acc"] = ner_seq_scores["overall_f1"]
        
        return outputs

    trainer = MultitaskTrainer(
        model=model,
        args=TrainingArguments(
            output_dir="./models/multitask_model",
            overwrite_output_dir=True,
            learning_rate=1e-5,
            do_train=True,
            num_train_epochs=3,
            # Adjust batch size if this doesn't fit on the Colab GPU
            per_device_train_batch_size=16,
            per_device_eval_batch_size=4,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            # By default, Trainer disallow saving weights contained shared tensors, therefore setting this to False
            save_safetensors=False,
        ),
        data_collator=GrabberBertDataCollator(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def main():
    
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    max_length = 128

    multitask_model = Multitask_BERT_v2.create(
        model_name=model_name,
        model_type_dict={
            "intent": BertForSequenceClassification,
            "ner": BertForTokenClassification,
        },
        model_config_dict={
            "intent": BertConfig.from_pretrained(model_name, num_labels=len(INTENT_LABEL)),
            "ner": BertConfig.from_pretrained(model_name, num_labels=len(NER_LABEL)),
        }
    )

    evaluators = {
        "f1_metric": evaluate.load("f1"),
        "seqeval": evaluate.load("seqeval"),
    }

    ds = call_dataset()

    # Prepare datasets for Cantonese
    for lang in ds.datasets.keys():
    # for lang in ["English"]:
        # Train the model on each dataset
        ds.set_splits_by_lang(lang)
        train_ds, valid_ds = prepare_datasets(ds, tokenizer, max_length)
        train_mtmodel(multitask_model, tokenizer, evaluators, train_ds, valid_ds)

    multitask_model.push_to_hub(
        "borisPMC/MedicGrabber_multitask_BERT", 
        commit_message="Uploading Multitask BERT model",
        safe_serialization=False)
    tokenizer.push_to_hub("borisPMC/MedicGrabber_multitask_BERT")

    print("Uploaded, exiting...")
    

if __name__ == "__main__":
    main()