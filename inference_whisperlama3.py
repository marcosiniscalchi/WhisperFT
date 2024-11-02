from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from torch.utils.data import DataLoader
from data_classes.whisper_dataset import TextProcessingForLlama
from data_classes.whisper_dataset import WhisperLlamaDataset
from transformers import AutoTokenizer
import os
from yaml_config_override import add_arguments
from addict import Dict as ConfDict
from datasets import load_dataset
import torch
from typing import Any
from evaluate import load
from tqdm import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load configuration from yaml file
config = add_arguments()
config = ConfDict(config)




def test(model: Any, test_loader: Any, llama_tokenizer: Any) :

    model.eval()
    predictions = []
    references  = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    wer = load("wer")
    
    for batch in tqdm(test_loader, desc="Evaluating"):
    #for batch_idx, batch in enumerate(test_loader):
        input_features = batch["input_features"]
        ref_labels     = batch["transcriptions"]
        attention_mask = batch["attention_mask"]
        with torch.no_grad():
            predicted_ids =  model.generate(batch["input_features"].to(device), 
                                            attention_mask=batch["attention_mask"].to(device),
                                            language=config.inference.inference_language)
        pred_str = llama_tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        #pred_ref = processor.batch_decode(ref_labels, skip_special_tokens=True)
        pred_ref = ref_labels
        predictions.extend(pred_str)
        references.extend(pred_ref)
    
    print(100 * wer.compute(references=references, predictions=predictions))
    
    
if __name__ == "__main__":
    path_to_model=config.inference.inference_model_tag_or_path    
    processor = WhisperProcessor.from_pretrained(config.processor.processor_tag_or_path, 
                                                 language=config.processor.language, 
                                                 task=config.processor.task)
    
    model = WhisperForConditionalGeneration.from_pretrained(path_to_model)
    
    llama_tokenizer =  AutoTokenizer.from_pretrained(config.inference.inference_tokenizer_path, padding_side="right")
    
    ds = load_dataset(config.data.dataset, config.data.language, split=config.inference.inference_split)
    #temp_dataset= ds.train_test_split(test_size=0.99)
    #ds= temp_dataset["train"]
    ds            = ds.select_columns(['audio', 'sentence'])

    test_dataset = WhisperLlamaDataset(tf_dataset=ds, 
                                       llama_tokenizer=llama_tokenizer,
                                       model_language=config.tokenizer.tokenizer_language,
                                       model_task=config.tokenizer.tokenizer_task,
                                       sampling_rate=config.featextractor.featextractor_sampling_rate,
                                       max_length_in_second=config.featextractor.featextractor_max_length_in_second,
                                       padding=config.featextractor.featextractor_padding,
                                       return_tensor=config.featextractor.featextractor_return_tensor,
                                       return_attention_mask=config.featextractor.featextractor_return_attention_mask,
                                       )
            
     
    #item = test_dataset.__getitem__(3)
    from torch.utils.data import DataLoader
    data_processing = TextProcessingForLlama(
        processor=processor,
        llama_tokenizer=llama_tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        )
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.inference.inference_batch_size,
                             shuffle=False,
                             collate_fn=lambda x: data_processing(x),
                             )
    test(model, test_loader, llama_tokenizer)