# Source: https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb#scrollTo=8326221e-ec13-4731-bb4e-51e5fc1486c5
# Collator for Speech-to-Text

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, Wav2Vec2Processor
)
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    feature_extractor: Union[Any, WhisperFeatureExtractor]
    tokenizer: Union[Any, WhisperTokenizer]
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
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
