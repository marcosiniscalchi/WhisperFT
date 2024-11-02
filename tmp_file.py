from torch.utils.data import Dataset
from dataclasses import dataclass
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict, Audio
import torch
import librosa
from transformers import WhisperForConditionalGeneration


processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

ds = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test")

clip = ds[0]["audio"]["array"]
sample = librosa.resample(clip, orig_sr=48000, target_sr=16000)
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

inputs = feature_extractor(sample, return_attention_mask=True, sampling_rate=16000, return_tensor="pt", padding="max_length", max_length=30*16000)
x = torch.tensor(inputs.input_features)

text_input = ["बराईशी हिन्सा पूलीस की भूमीका सन्दिक्त बाईरेल रूएत स्वीरें"]
decoder_input_ids = processor.tokenizer(text_input, return_tensors="pt").input_ids

import torch.nn.functional as F
model.eval()
with torch.no_grad():
    out_whisper = model(x, decoder_input_ids=decoder_input_ids)

prob_whisper = F.softmax(out_whisper.logits, dim=-1)
prdict_ids = torch.argmax(prob_whisper, dim=-1)
print(processor.batch_decode(prdict_ids)[0])

from transformers import AutoTokenizer

llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", padding_side="right")
model_lama = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model_lama.config.vocab_size = 128256
model_lama.config.decoder_start_token_id=128000
model_lama.config.pad_token_id = 128002
model_lama.config.decoder_pad_token_id=128002

padding_idx=128002
embedding = torch.nn.Embedding(128256, 768, padding_idx=padding_idx, scale_grad_by_freq=False, sparse=False)
from torch.nn import Linear
model_lama.eval()
with torch.no_grad():
    model_lama.proj_out=Linear(in_features=768, out_features=128256, bias=False)
    model_lama.model.decoder.embed_tokens = embedding
    out_whisper_llama = model_lama(x, decoder_input_ids=decoder_input_ids_llama3)

prob_whisper_llama = F.softmax(out_whisper_llama.logits, dim=-1)
prdict_ids_llama = torch.argmax(prob_whisper_llama, dim=-1)
print(llama3_tokenizer.batch_decode(prdict_ids_llama)[0])

    

decoder_input_ids_llama3 = llama3_tokenizer(text_input, return_tensors="pt").input_ids
#llama3_tokenizer.decode(decoder_input_ids_llama3[0].tolist())

# Add whisper special token into llama special token
new_tokens = processor.tokenizer.all_special_tokens
llama3_tokenizer.add_special_tokens({"additional_special_tokens" : new_tokens})
new_vocabulary_size = len(llama3_tokenizer)
model_lama.resize_token_embeddings(new_vocabulary_size)
# To know the id for a new added special token
llama3_tokenizer.convert_tokens_to_ids('<|startoftranscript|>')

# Typical sentence structure sent to whisper decoder as decoder_inputs_ids: 
#'<|startoftranscript|><|hi|><|transcribe|><|notimestamps|>वह टेनिस बहुत अच्छा खेलती है।<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>'




