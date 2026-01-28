# Qwen3-TTS Fine-Tuning Studio

A complete toolkit for fine-tuning **Qwen3-TTS** models with custom voices. This repository includes scripts for data preparation, training, sample generation, and an interactive web demo for comparing checkpoints.

## 🎯 Features

- **Data Preparation**: Format your audio dataset and extract audio codes
- **Fine-Tuning**: Train custom voice models with configurable hyperparameters
- **Sample Generation**: Generate samples from all training checkpoints
- **Web Demo**: Compare voice quality across epochs side-by-side

## 📁 Repository Structure

```
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── prepare_data.py               # Extract audio codes from raw data
├── dataset.py                    # Dataset class for training
├── sft_12hz.py                   # Main fine-tuning script
├── generate_comparison_samples.py # Generate samples from all checkpoints
├── serve_comparison.py           # Web demo for checkpoint comparison
├── webui.py                      # Gradio web UI (optional)
└── scripts/
    └── format_dataset.py         # Format raw dataset to JSONL
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install qwen-tts librosa soundfile
```

### 2. Prepare Your Dataset

Create a directory with your audio files and a metadata file:

```
my_dataset/
├── metadata.csv      # Format: filename|transcript
└── wavs/
    ├── audio_001.wav
    └── audio_002.wav
```

Format the dataset:

```bash
python scripts/format_dataset.py \
  --dataset_dir /path/to/my_dataset \
  --output_dir ./my_finetune_data \
  --ref_audio /path/to/my_dataset/wavs/audio_001.wav
```

### 3. Extract Audio Codes

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl ./my_finetune_data/train_raw.jsonl \
  --output_jsonl ./my_finetune_data/train_with_codes.jsonl
```

### 4. Fine-Tune the Model

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path ./output_model \
  --train_jsonl ./my_finetune_data/train_with_codes.jsonl \
  --batch_size 1 \
  --lr 5e-6 \
  --num_epochs 20 \
  --grad_accum 8 \
  --speaker_name my_voice \
  --export_mode custom_voice \
  --speaker_index 3000 \
  --attn_impl sdpa
```

### 5. Generate Comparison Samples

```bash
python generate_comparison_samples.py
```

### 6. Launch Web Demo

```bash
python serve_comparison.py
# Open http://localhost:8890 in your browser
```

## 📊 Training Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--lr` | Learning rate | `5e-6` for stability |
| `--num_epochs` | Number of epochs | `15-20` |
| `--batch_size` | Batch size | `1-2` (GPU dependent) |
| `--grad_accum` | Gradient accumulation steps | `8` |
| `--export_mode` | Export format | `custom_voice` |
| `--attn_impl` | Attention implementation | `sdpa` or `flash_attention_2` |

## 🔊 Model Variants

- **0.6B-Base**: Smaller, faster inference
- **1.7B-Base**: Better quality, requires more VRAM (~20GB)
- **CustomVoice**: Pre-trained with speaker embedding
- **VoiceDesign**: Supports voice description prompts


## 🔄 Resuming Training

To resume training:

### 1. From a Base Checkpoint
Use the standard command:
```bash
python sft_12hz.py --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base ...
```

### 2. From a CustomVoice Checkpoint (IMPORTANT)
If you are resuming from a checkpoint you exported with `--export_mode custom_voice`, you must provide the same speaker index and ensure the script treats it correctly (the `speaker_encoder` is removed in these checkpoints).

```bash
python sft_12hz.py \
  --init_model_path ./output_model/checkpoint-epoch-X \
  --output_model_path ./output_model \
  --export_mode custom_voice \
  --speaker_name my_voice \
  --speaker_index 3000 \
  ...
```

The script includes a fallback mechanism to load the speaker embedding from the codec weights if the `speaker_encoder` is missing.

## 🎧 Auto-Generate Samples

This repository includes tools to automatically generate and host samples as training progresses.

### Generate Spanish Samples
Generate 5 test phrases from the latest checkpoint:

```bash
python generate_spanish_samples.py --latest
```

Or target a specific checkpoint:
```bash
python generate_spanish_samples.py --checkpoint ./output/checkpoint-epoch-5
```

### Continuous Monitoring
Run the watcher script to monitor your output directory. It will detect new checkpoints and generate samples automatically:

```bash
python watch_and_generate.py
```

Results are saved to `spanish_samples/` with an `index.html` gallery for easy listening.

## 📝 License

This project follows the Qwen3-TTS license. See the [Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS) for details.

## 🙏 Acknowledgments

- [Qwen Team](https://github.com/QwenLM) for the Qwen3-TTS model
- [Hugging Face](https://huggingface.co) for Transformers and Accelerate
