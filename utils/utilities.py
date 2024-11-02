from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

# HF datacollator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

# HF datacollator
@dataclass
class LlamaDataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    llama_tokenizer: Any = None
    decoder_start_token_id: int = 128000
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.llama_tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

# HF datacollator
@dataclass
class LlamaDataCollatorSpeechSeq2SeqWithPaddingWhisperSpecialTokens:
    processor: Any
    llama_tokenizer: Any = None
    decoder_start_token_id: int = 128000
    language: str = "<|hi|>"
    task:   str = "<|transcribe|>"
    timestamp: str = "<|notimestamps|>"
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.llama_tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # replace bos_token_id with sot_token_id
        labels = labels.masked_fill(labels.eq(self.llama_tokenizer.bos_token_id), self.decoder_start_token_id)
        lang_id = self.llama_tokenizer.convert_tokens_to_ids(self.language)
        task_id = self.llama_tokenizer.convert_tokens_to_ids(self.task)
        timestamp_id = self.llama_tokenizer.convert_tokens_to_ids(self.timestamp)
        
        header = [self.decoder_start_token_id, lang_id, task_id, timestamp_id]
        temporary_labels = [item.tolist() for item in [lab[1:] for lab in labels]]
        labels = [header + tail for tail in temporary_labels]
        labels = torch.as_tensor(labels)
        
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch





# Compute metrics
def prepare_compute_metrics(tokenizer, metric, pad_token_id):        
    def compute_metrics(pred):        
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # replace -100 with the pad_token_id
        ##tokenizer.pad_token_id = 128002 # remove it if whisper no longer works
        label_ids[label_ids == pad_token_id] =  tokenizer.pad_token_id
        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    return compute_metrics
