
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

Now start the training loop.

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

**Parameters:**
-   `--batch_size`: Adjust based on your GPU VRAM.
-   `--num_epochs`: 10-20 epochs is usually good for small datasets (~100 samples).

## 4. Inference / Verification

Once training is done, you can test your new model.

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Load your fine-tuned checkpoint (e.g., epoch 9)
checkpoint_path = "./finetuning/output_model/checkpoint-epoch-9"

model = Qwen3TTSModel.from_pretrained(
    checkpoint_path,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Generate speech
wavs, sr = model.generate_custom_voice(
    text="This is my new cloned voice speaking.",
    speaker="my_custom_voice",  # Must match the name used in training
)

sf.write("output_test.wav", wavs[0], sr)
```
