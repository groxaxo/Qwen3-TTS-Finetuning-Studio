
# Qwen3-TTS Fine-Tuning Guide (Detailed)

This guide provides a comprehensive walkthrough for fine-tuning **Qwen3-TTS-12Hz-1.7B/0.6B-Base** models. We have added an automated script to simplify dataset preparation.

## 0. Prerequisites

Ensure you have installed the package and dependencies:

```bash
# Basic install
pip install qwen-tts

# Install extra dependencies for our formatting script
pip install librosa soundfile
```

Clone the repository if you haven't:

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
```

## 1. Prepare Your Dataset

You need a dataset consisting of:
1.  **Audio files**: A collection of `.wav` files (any sample rate, our script will resample them).
2.  **Metadata**: A text file (csv or pipe-separated) mapping filenames to transcripts.

**Format Example (`metadata.csv`):**
```text
utt_001|Hello world.
utt_002|This is a test.
```

Your directory structure should look like this:
```
my_dataset/
├── metadata.csv
└── wavs/
    ├── utt_001.wav
    └── utt_002.wav
```
*(Note: audio files can also be in the root of `my_dataset/`)*

### Automatic Formatting

We provide a script `scripts/format_dataset.py` that:
-   Resamples all audio to **24000 Hz** (Required!).
-   Generates the `train_raw.jsonl` file.
-   Handles reference audio.

**Usage:**

```bash
# Run from the root of Qwen3-TTS
python scripts/format_dataset.py \
  --dataset_dir /path/to/my_dataset \
  --output_dir ./my_finetune_data \
  --ref_audio /path/to/my_dataset/wavs/utt_001.wav
```

> [!TIP]
> **Reference Audio (`--ref_audio`)**: Qwen3-TTS uses a reference speaker embedding. We heavily recommend using **one consistent audio file** as the reference for all training samples to ensure stability. The script will handle this for you.

After running this, `./my_finetune_data` will contain:
-   `train_raw.jsonl`
-   `wavs/` (re-sampled to 24k)
-   `ref_audio.wav`

## 2. Extract Audio Codes

Now we convert the raw audio into discrete codes used by the model.

```bash
cd finetuning

python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl ../my_finetune_data/train_raw.jsonl \
  --output_jsonl ../my_finetune_data/train_with_codes.jsonl
```

## 3. Run Fine-Tuning (SFT)

### Option A: Using the Fixed Training Script (Recommended)

We provide `sft_12hz_fixed_export.py` which fixes critical bugs and supports both export modes:

**For CustomVoice Export (simplest inference):**
```bash
python sft_12hz_fixed_export.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path ./output_model \
  --train_jsonl ../my_finetune_data/train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 10 \
  --speaker_name my_custom_voice \
  --export_mode custom_voice \
  --speaker_index 3000 \
  --attn_impl sdpa
```

**For Base-Style Export (keeps voice cloning capability):**
```bash
python sft_12hz_fixed_export.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path ./output_model \
  --train_jsonl ../my_finetune_data/train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 10 \
  --speaker_name my_custom_voice \
  --export_mode base \
  --attn_impl sdpa
```

**Key Parameters:**
-   `--export_mode`: Choose `custom_voice` (bakes speaker embedding, simpler inference) or `base` (keeps speaker encoder for voice cloning)
-   `--batch_size`: Adjust based on your GPU VRAM
-   `--num_epochs`: 10-20 epochs is usually good for small datasets (~100 samples)
-   `--grad_accum`: Gradient accumulation steps (default: 4)
-   `--attn_impl`: Attention implementation (`sdpa` is safer than `flash_attention_2`)
-   `--speaker_index`: Index for baking speaker embedding (only for custom_voice mode, default: 3000)

### Option B: Using the Original Script

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path ./output_model \
  --train_jsonl ../my_finetune_data/train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 10 \
  --speaker_name my_custom_voice
```

> **Note:** The original script has known bugs (see `sft_12hz_fixed_export.py` for fixes). Use the fixed version for better results.

## 4. Inference / Verification

### CustomVoice Mode (if using --export_mode custom_voice)

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Load your fine-tuned checkpoint (e.g., epoch 9)
checkpoint_path = "./output_model/checkpoint-epoch-9"

model = Qwen3TTSModel.from_pretrained(
    checkpoint_path,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

# Get the speaker name
speaker = model.get_supported_speakers()[0]

# Generate speech
wavs, sr = model.generate_custom_voice(
    text="This is my new cloned voice speaking.",
    language="Auto",
    speaker=speaker,  # Use the speaker from training
)

sf.write("output_test.wav", wavs[0], sr)
```

### Base Mode (if using --export_mode base)

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Load your fine-tuned checkpoint
checkpoint_path = "./output_model/checkpoint-epoch-9"

model = Qwen3TTSModel.from_pretrained(
    checkpoint_path,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

# Generate speech using voice cloning
wavs, sr = model.generate_voice_clone(
    text="This is my new cloned voice speaking.",
    language="Auto",
    ref_audio="../my_finetune_data/ref_audio.wav",
)

sf.write("output_test.wav", wavs[0], sr)
```
