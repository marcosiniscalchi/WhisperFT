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

from data_classes.whisper_dataset import WhisperDataset, TextProcessing
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from utils.utilities import prepare_compute_metrics
from utils.utilities import DataCollatorSpeechSeq2SeqWithPadding

# Load configuration from yaml file
config = add_arguments()
config = ConfDict(config)

def train(dataset_tr,dataset_ts, metric, pad_token_id):
    
    # Load model to be finetuned
    model = WhisperForConditionalGeneration.from_pretrained(config.model.model_tag)
    
    # Load tokenizer for specific language and task
    tokenizer = WhisperTokenizer.from_pretrained(config.tokenizer.tokenizer_tag_or_path, 
                                                 language=config.tokenizer.tokenizer_language, 
                                                 task=config.tokenizer.tokenizer_task)
    
    processor = WhisperProcessor.from_pretrained(config.tokenizer.tokenizer_tag_or_path, 
                                                 language=config.tokenizer.tokenizer_language, 
                                                 task=config.tokenizer.tokenizer_task)
    # Wrap HP datasat in a torch dataset
    ds  = WhisperDataset(tf_dataset=dataset_tr, 
                         tokenizer=config.tokenizer.tokenizer_tag_or_path,
                         model_language=config.tokenizer.tokenizer_language,
                         model_task=config.tokenizer.tokenizer_task,
                         sampling_rate=config.featextractor.featextractor_sampling_rate,
                         max_length_in_second=config.featextractor.featextractor_max_length_in_second,
                         padding=config.featextractor.featextractor_padding,
                         return_tensor=config.featextractor.featextractor_return_tensor,
                         return_attention_mask=config.featextractor.featextractor_return_attention_mask,
                         )
    ds_ts = WhisperDataset(tf_dataset=dataset_ts, 
                           tokenizer=config.tokenizer.tokenizer_tag_or_path,
                           model_language=config.tokenizer.tokenizer_language,
                           model_task=config.tokenizer.tokenizer_task,
                           sampling_rate=config.featextractor.featextractor_sampling_rate,
                           max_length_in_second=config.featextractor.featextractor_max_length_in_second,
                           padding=config.featextractor.featextractor_padding,
                           return_tensor=config.featextractor.featextractor_return_tensor,
                           return_attention_mask=config.featextractor.featextractor_return_attention_mask,
                           )

    
    # A datacollator is needed to wrap-up bached of data
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
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
    
    compute_metrics = prepare_compute_metrics(tokenizer, metric, pad_token_id)
    
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
    
    # Load HF dataset -- common voice in this case
    common_voice_ts = load_dataset(config.data.dataset, config.data.language, split="test")
    common_voice_ts = common_voice_ts.select_columns(["audio", "sentence"])
    
    metric = evaluate.load("wer")
    pad_token_id = -100
    
    train(common_voice_tr, 
          common_voice_ts, 
          metric, 
          pad_token_id)
    
    

    







