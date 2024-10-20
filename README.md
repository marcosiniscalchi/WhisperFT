# WhisperFT
# Exploring Flan-T5 for Post-ASR Error Correction

This repository contains the code and experiments for fine-tuning Whisper w and w/ LoRA. The project is meant for learning the basic concept for Foundation Models.

> [!IMPORTANT]
> All the code for running the experiments is available in this repository. The pre-trained models are available on the Hugging Face model hub. [

## How to run the code

This repository contains the code for training and evaluating the FLANEC models. Before starting, you may want to install the required dependencies:

```bash
pip install -r requirements.txt
```

### Training and Evaluation

The `config/` folder contains the configuration files for training and evaluation of all model types (e.g., standard-ft and LoRA).

To train a model, you may want to look at the specific configuration file:

- `config/whisper_small.yaml` for the small model with standard fine-tuning 🔥
- `config/whisper_small_lora.yaml` for the small model with LoRA fine-tuning 🔥

To adapt a model, run the following command:

- **LoRA** - small model example:

```bash
python train_whisper_lora.py --config config/whisper_small_lora.yaml
```

- **Standard fine-tuning** - base model example:

```bash
python train_whisper.py --config config/whisper_small.yaml
```

To evaluate a model, you may want to look at the specific configuration file and then run the following command:

- **LoRA** - base model example:

```bash
python infer_whisper_lora.py --config config/whisper_small_lora.yaml
```

- **Standard fine-tuning** - base model example:

```bash
python infer_whisper.py --config config/whisper_small.yaml
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
