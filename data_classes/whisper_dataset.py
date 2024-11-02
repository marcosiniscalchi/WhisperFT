from torch.utils.data import Dataset
from dataclasses import dataclass
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict, Audio
import torch
import librosa


@dataclass
class TextProcessing():
    processor: Any
    decoder_start_token_id: int
    
    def __call__(self, features):
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
        batch["attention_mask"] = torch.asarray([feature["attention_mask"] for feature in features]) #[{"attention_mask": feature["attention_mask"]} for feature in features]
        batch["transcriptions"] = [ feature["transcriptions"] for feature in features]
        return batch

@dataclass
class TextProcessingForLlama():
    processor: Any = None
    llama_tokenizer: Any = None
    decoder_start_token_id: int = 128000
    
    def __call__(self, features):
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
        batch["attention_mask"] = torch.asarray([feature["attention_mask"] for feature in features]) #[{"attention_mask": feature["attention_mask"]} for feature in features]
        batch["transcriptions"] = [ feature["transcriptions"] for feature in features]
        return batch


@dataclass
class WhisperDataset(Dataset):
    '''
        This is a wrapper for the whisper data in huggingface
    '''
    model_tag: str = "openai/whisper-small"
    model_language: str = "hi"
    model_task: str = "transcribe"
    feature_extractor_name_or_path: str = "openai/whisper-small"
    tokenizer_name_or_path: str = "openai/whisper-small"
    max_length: int = 1024
    decoder_start_token_id: int = -1
    feature_extractor: Any = None
    tokenizer: Any = None
    sampling_rate: int = 16000
    tf_dataset: Any = None
    max_length_in_second: int = 30
    padding : str = "max_length"
    return_tensor: str = "pt"
    return_attention_mask: bool = True
     
    def __post_init__(self):
        self.feature_extractor=WhisperFeatureExtractor.from_pretrained(self.model_tag)
        self.tokenizer = WhisperTokenizer.from_pretrained(self.model_tag, language=self.model_language, task=self.model_task)
        self.ds = self.tf_dataset 
        
        
    
    def _resample(self, clip, sampling_rate):
        clip = librosa.resample(clip, orig_sr=sampling_rate, target_sr=self.sampling_rate)
        return clip
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):

        sample = self.ds[idx]
        clip = self._resample(sample["audio"]["array"],sampling_rate=sample["audio"]["sampling_rate"])
        
        
        inputs  = self.feature_extractor(clip, 
                                         return_attention_mask=self.return_attention_mask, 
                                         sampling_rate=self.sampling_rate, 
                                         return_tensor=self.return_tensor,
                                         padding=self.padding,
                                         max_length=self.max_length_in_second*self.sampling_rate)
        input_ids  = self.tokenizer(sample["sentence"]).input_ids
        transcript  = sample["sentence"]

        item = {
            "input_features": inputs["input_features"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels"         : input_ids,
            "transcriptions" : transcript,
        }
        return item


@dataclass
class WhisperLlamaDataset(Dataset):
    '''
        This is a wrapper for the whisper data in huggingface
    '''
    model_tag: str = "openai/whisper-small"
    model_language: str = "hi"
    model_task: str = "transcribe"
    feature_extractor_name_or_path: str = "openai/whisper-small"
    tokenizer_name_or_path: str = "openai/whisper-small"
    max_length: int = 1024
    decoder_start_token_id: int = -1
    feature_extractor: Any = None
    #tokenizer: Any = None
    llama_tokenizer: Any = None
    sampling_rate: int = 16000
    tf_dataset: Any = None
    max_length_in_second: int = 30
    padding : str = "max_length"
    return_tensor: str = "pt"
    return_attention_mask: bool = True
     
    def __post_init__(self):
        self.feature_extractor=WhisperFeatureExtractor.from_pretrained(self.feature_extractor_name_or_path)
        self.ds = self.tf_dataset 
        
        
    
    def _resample(self, clip, sampling_rate):
        clip = librosa.resample(clip, orig_sr=sampling_rate, target_sr=self.sampling_rate)
        return clip
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):

        sample = self.ds[idx]
        clip = self._resample(sample["audio"]["array"],sampling_rate=sample["audio"]["sampling_rate"])
        
        
        inputs  = self.feature_extractor(clip, 
                                         return_attention_mask=self.return_attention_mask, 
                                         sampling_rate=self.sampling_rate, 
                                         return_tensor=self.return_tensor,
                                         padding=self.padding,
                                         max_length=self.max_length_in_second*self.sampling_rate)
        
        input_ids = self.llama_tokenizer(sample["sentence"]).input_ids # llama tokenizer used to obtain llama's ids
        transcript  = sample["sentence"]

        item = {
            "input_features": inputs["input_features"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels"         : input_ids,
            "transcriptions" : transcript,
        }
        return item

      
@dataclass
class WhisperLlamaDatasetWhisperSpecialTokens(Dataset):
    model_tag: str = "openai/whisper-small"
    model_language: str = "hi"
    model_task: str = "transcribe"
    feature_extractor_name_or_path: str = "openai/whisper-small"
    tokenizer_name_or_path: str = "openai/whisper-small"
    max_length: int = 1024
    decoder_start_token_id: int = -1
    feature_extractor: Any = None
    #tokenizer: Any = None
    llama_tokenizer: Any = None
    sampling_rate: int = 16000
    tf_dataset: Any = None
    max_length_in_second: int = 30
    padding : str = "max_length"
    return_tensor: str = "pt"
    return_attention_mask: bool = True
     
    def __post_init__(self):
        self.feature_extractor=WhisperFeatureExtractor.from_pretrained(self.model_tag)
        #self.tokenizer = WhisperTokenizer.from_pretrained(self.model_tag, language=self.model_language, task=self.model_task)
        self.ds = self.tf_dataset 
        
        
    
    def _resample(self, clip, sampling_rate):
        clip = librosa.resample(clip, orig_sr=sampling_rate, target_sr=self.sampling_rate)
        return clip
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):

        sample = self.ds[idx]
        clip = self._resample(sample["audio"]["array"],sampling_rate=sample["audio"]["sampling_rate"])
        
        
        inputs  = self.feature_extractor(clip, 
                                         return_attention_mask=self.return_attention_mask, 
                                         sampling_rate=self.sampling_rate, 
                                         return_tensor=self.return_tensor,
                                         padding=self.padding,
                                         max_length=self.max_length_in_second*self.sampling_rate)
        # input_ids  = self.tokenizer(sample["sentence"]).input_ids
        
        input_ids = self.llama_tokenizer(sample["sentence"]).input_ids # llama tokenizer used to obtain llama's ids
        input_ids.append(self.llama_tokenizer.pad_token_id)
        ##print(sample["sentence"])
        transcript  = sample["sentence"]

        item = {
            "input_features": inputs["input_features"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels"         : input_ids,
            "transcriptions" : transcript,
        }
        return item
 
if __name__ == "__main__":
    from tqdm import tqdm

    path_to_model="/home/marco/FlanEC/whisper-small-hi_marco/checkpoint-500"
    from transformers import WhisperForConditionalGeneration
    
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(path_to_model)
    
    ds = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test")
    #temp_dataset= ds.train_test_split(test_size=0.99)
    #ds= temp_dataset["train"]
    ds            = ds.select_columns(['audio', 'sentence'])

    test_dataset = WhisperDataset(tf_dataset=ds, 
                                  model_tag="openai/whisper-small", 
                                  model_language="hi",
                                  model_task="transcribe", 
                                  sampling_rate=16000)
    
    #item = test_dataset.__getitem__(3)
    from torch.utils.data import DataLoader
    data_processing = TextProcessing(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        )
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=16,
                             shuffle=False,
                             collate_fn=lambda x: data_processing(x),
                             )
    
    model.eval()
    predictions = []
    references  = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    from evaluate import load
    wer = load("wer")
    
    for batch in tqdm(test_loader, desc="Evaluating"):
    #for batch_idx, batch in enumerate(test_loader):
        input_features = batch["input_features"]
        ref_labels     = batch["transcriptions"]
        attention_mask = batch["attention_mask"]
        with torch.no_grad():
            predicted_ids =  model.generate(batch["input_features"].to(device), 
                                            attention_mask=batch["attention_mask"].to(device),
                                            language="hi")
        pred_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        #pred_ref = processor.batch_decode(ref_labels, skip_special_tokens=True)
        pred_ref = ref_labels
        predictions.extend(pred_str)
        references.extend(pred_ref)
    
    print(100 * wer.compute(references=references, predictions=predictions))