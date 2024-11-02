import torch
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import WhisperTokenizer
import datasets
from datasets import load_dataset

from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import WhisperForConditionalGeneration
import os
import evaluate
from yaml_config_override import add_arguments
from addict import Dict as ConfDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from data_classes.whisper_dataset import WhisperLlamaDataset
from utils.utilities import prepare_compute_metrics
from utils.utilities import LlamaDataCollatorSpeechSeq2SeqWithPadding
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import Linear
from torch.nn import ModuleList

os.environ["TOKENIZERS_PARALLELISM"] = "true"



# Load configuration from yaml file
config = add_arguments()
config = ConfDict(config)

def train(dataset_tr,dataset_ts, metric, pad_token_id):
    
    # Load llama3 model and tokenizer
    #llama3_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", padding_side="right")

    # Load whisper processor for grabbing the special tokens
    processor = WhisperProcessor.from_pretrained(config.tokenizer.tokenizer_tag_or_path, 
                                                 language=config.tokenizer.tokenizer_language, 
                                                 task=config.tokenizer.tokenizer_task)
    # Load whisper model to be finetuned
    model = WhisperForConditionalGeneration.from_pretrained(config.model.model_tag)
        
    new_vocabulary_size = len(llama3_tokenizer)
    model.resize_token_embeddings(new_vocabulary_size)
    model.config.decoder_start_token_id = llama3_tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    model.config.pad_token_id = llama3_tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")
    model.config.decoder_pad_token_id = llama3_tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")
    model.config.vocab_size = new_vocabulary_size
    embedding_dim = model.model.decoder.embed_tokens.embedding_dim
    llama3_tokenizer.pad_token_id = llama3_tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")
    llama3_tokenizer.save_pretrained(config.training.training_output_dir + "/tokenizer/")

    
    model.eval()
    with torch.no_grad():
        model.model.decoder.embed_tokens.padding_idx = llama3_tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")
        #model.proj_out=Linear(in_features=embedding_dim, out_features=new_vocabulary_size, bias=False)
   
    # Freeze encoder
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    
    model.train()
    # Wrap HP datasat in a torch dataset
    ds  = WhisperLlamaDataset(tf_dataset=dataset_tr, 
                              llama_tokenizer=llama3_tokenizer,
                              model_language=config.tokenizer.tokenizer_language,
                              model_task=config.tokenizer.tokenizer_task,
                              sampling_rate=config.featextractor.featextractor_sampling_rate,
                              max_length_in_second=config.featextractor.featextractor_max_length_in_second,
                              padding=config.featextractor.featextractor_padding,
                              return_tensor=config.featextractor.featextractor_return_tensor,
                              return_attention_mask=config.featextractor.featextractor_return_attention_mask,
                              )
    ds_ts = WhisperLlamaDataset(tf_dataset=dataset_ts, 
                                llama_tokenizer=llama3_tokenizer,
                                model_language=config.tokenizer.tokenizer_language,
                                model_task=config.tokenizer.tokenizer_task,
                                sampling_rate=config.featextractor.featextractor_sampling_rate,
                                max_length_in_second=config.featextractor.featextractor_max_length_in_second,
                                padding=config.featextractor.featextractor_padding,
                                return_tensor=config.featextractor.featextractor_return_tensor,
                                return_attention_mask=config.featextractor.featextractor_return_attention_mask,
                                )

    
    # A datacollator is needed to wrap-up bached of data
    data_collator = LlamaDataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        llama_tokenizer=llama3_tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id, 
        )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.training.training_output_dir,
        per_device_train_batch_size=config.training.training_per_device_train_batch_size,
        gradient_accumulation_steps=config.training.training_gradient_accumulation_steps,
        learning_rate=config.training.training_learning_rate, # 1e-5
        warmup_steps=config.training.training_warmup_steps,
        max_steps=config.training.training_max_steps,
        gradient_checkpointing=config.training.training_gradient_checkpointing,
        fp16=config.training.training_fp16,
        evaluation_strategy=config.training.training_evaluation_strategy,
        save_strategy=config.training.training_save_strategy,
        per_device_eval_batch_size=config.training.training_per_device_eval_batch_size,
        predict_with_generate=config.training.training_predict_with_generate,
        generation_max_length=config.training.training_generation_max_length,
        save_steps=config.training.training_save_steps,
        eval_steps=config.training.training_eval_steps,
        logging_steps=config.training.training_logging_steps,
        report_to=config.training.training_report_to,
        run_name=config.training.training_run_name,
        load_best_model_at_end=config.training.training_load_best_model_at_end,
        metric_for_best_model=config.training.training_metric_for_best_model,
        greater_is_better=config.training.training_greater_is_better,
        push_to_hub=config.training.training_push_to_hub,
        remove_unused_columns=config.training.training_remove_unused_columns,
        dataloader_num_workers=config.training.training_dataloader_num_workers,
        )
    
    
    compute_metrics = prepare_compute_metrics(llama3_tokenizer, metric, pad_token_id)
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds,
        eval_dataset=ds_ts,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        )
    
    trainer.train()
    

    
if __name__ == "__main__":
    
    # Load HF dataset -- common voice in this case
    common_voice_tr = load_dataset(config.data.dataset, config.data.language, split=config.data.split)
    common_voice_tr = common_voice_tr.select_columns(["audio", "sentence"])
    temp_dataset= common_voice_tr.train_test_split(test_size=0.99)
    common_voice_tr= temp_dataset["train"]
    
    # Load HF dataset -- common voice in this case
    common_voice_ts = load_dataset(config.data.dataset, config.data.language, split="test")
    common_voice_ts = common_voice_ts.select_columns(["audio", "sentence"])
    temp_dataset= common_voice_ts.train_test_split(test_size=0.99)
    common_voice_ts= temp_dataset["train"]
   
    
    metric = evaluate.load("wer")
    pad_token_id = -100
    
    train(common_voice_tr, 
          common_voice_ts, 
          metric, 
          pad_token_id)
    