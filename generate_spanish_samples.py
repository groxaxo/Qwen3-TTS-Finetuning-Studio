#!/usr/bin/env python3
"""Generate 5 Spanish samples from a specific checkpoint."""

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import os
import json
from datetime import datetime
import gc
import glob
import argparse
import sys

# Configuration defaults
DEFAULT_CKPT_BASE = "output_model_monica_17b_v4"
OUTPUT_DIR = "spanish_samples"
SPEAKER_NAME = "monica"

# 5 Spanish phrases
TEST_PHRASES = [
    {
        "id": 1,
        "text": "Hola, esto es una prueba de síntesis de voz en español usando el último checkpoint.",
        "category": "Intro",
    },
    {
        "id": 2,
        "text": "La inteligencia artificial está transformando la manera en que interactuamos con la tecnología.",
        "category": "Tech",
    },
    {
        "id": 3,
        "text": "Me encanta caminar por el parque en una tarde soleada de primavera.",
        "category": "Casual",
    },
    {
        "id": 4,
        "text": "Por favor, confirma si la calidad del audio ha mejorado con el nuevo entrenamiento.",
        "category": "Question",
    },
    {
        "id": 5,
        "text": "Un viaje de mil millas comienza con un solo paso.",
        "category": "Proverb",
    },
]


def get_epoch(path):
    try:
        return int(path.split("-")[-1])
    except ValueError:
        return -1


def update_index_html(output_dir, new_results=None):
    """Regenerate index.html by scanning the output directory."""
    html_header = """
    <html>
    <head>
        <title>Spanish Samples Gallery</title>
        <style>
            body { font-family: sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; background: #f0f2f5; }
            h1 { color: #1a1a1a; }
            .epoch-section { background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .epoch-header { border-bottom: 2px solid #eee; padding-bottom: 1rem; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; }
            .samples-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; }
            .sample-card { background: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #e9ecef; }
            .sample-text { font-size: 0.9rem; color: #444; margin-bottom: 0.5rem; line-height: 1.4; }
            audio { width: 100%; margin-top: 0.5rem; }
            .meta { font-size: 0.8rem; color: #666; }
        </style>
    </head>
    <body>
        <h1>Spanish Training Progress</h1>
        <p>Gallery of generated samples across epochs.</p>
    """

    # Scan for all generated wav files
    wav_files = sorted(glob.glob(os.path.join(output_dir, "*.wav")))

    # Group by epoch
    samples_by_epoch = {}
    for wav in wav_files:
        basename = os.path.basename(wav)
        # Expected format: sample_XX_epochY.wav
        parts = basename.replace(".wav", "").split("_epoch")
        if len(parts) != 2:
            continue

        try:
            epoch = int(parts[1])
            if epoch not in samples_by_epoch:
                samples_by_epoch[epoch] = []
            samples_by_epoch[epoch].append(basename)
        except ValueError:
            continue

    html_body = ""
    # Sort epochs descending
    for epoch in sorted(samples_by_epoch.keys(), reverse=True):
        files = sorted(samples_by_epoch[epoch])
        html_body += f"""
        <div class="epoch-section">
            <div class="epoch-header">
                <h2>Epoch {epoch}</h2>
            </div>
            <div class="samples-grid">
        """

        for wav_file in files:
            # Match text if possible (assuming strict ordering or ID in filename)
            # Filename: sample_{id:02d}_epoch{epoch}.wav
            try:
                id_str = wav_file.split("_")[1]  # 01, 02...
                pid = int(id_str)
                text = next(
                    (p["text"] for p in TEST_PHRASES if p["id"] == pid), "Unknown text"
                )
                category = next(
                    (p["category"] for p in TEST_PHRASES if p["id"] == pid), "Sample"
                )
            except:
                text = wav_file
                category = "Unknown"

            html_body += f"""
                <div class="sample-card">
                    <div class="meta">{category}</div>
                    <div class="sample-text">"{text}"</div>
                    <audio controls src="{wav_file}"></audio>
                </div>
            """

        html_body += """
            </div>
        </div>
        """

    html_footer = f"""
        <p style="text-align: center; color: #666; margin-top: 3rem;">
            Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </p>
    </body>
    </html>
    """

    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_header + html_body + html_footer)

    print(f"✓ Updated index.html in {output_dir}")


def generate(checkpoint_path, epoch_num, output_dir):
    print("=" * 70)
    print(f"Generating samples for EPOCH {epoch_num} from {checkpoint_path}")
    print("=" * 70)

    try:
        print(f"Loading model...")
        tts = Qwen3TTSModel.from_pretrained(
            checkpoint_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        speakers = tts.get_supported_speakers()
        spk = SPEAKER_NAME if SPEAKER_NAME in speakers else speakers[0]
        print(f"Selected speaker: {spk}")

        for phrase in TEST_PHRASES:
            print(f"  Generating [{phrase['id']}/5]...")

            wavs, sr = tts.generate_custom_voice(
                text=phrase["text"],
                language="Spanish",
                speaker=spk,
                max_new_tokens=512,
            )

            filename = f"sample_{phrase['id']:02d}_epoch{epoch_num}.wav"
            filepath = f"{output_dir}/{filename}"
            sf.write(filepath, wavs[0], sr)

            print(f"       ✓ Saved {filename}")

        # Cleanup
        del tts
        torch.cuda.empty_cache()
        gc.collect()

        return True

    except Exception as e:
        print(f"❌ Error generating samples: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate Spanish samples from Qwen3-TTS checkpoint"
    )
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint path")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DEFAULT_CKPT_BASE,
        help="Base training directory to scan",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory for samples",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Automatically find latest checkpoint in base_dir",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    target_ckpt = None

    if args.checkpoint:
        target_ckpt = args.checkpoint
    elif args.latest:
        checkpoints = sorted(
            glob.glob(os.path.join(args.base_dir, "checkpoint-epoch-*"))
        )
        if checkpoints:
            target_ckpt = sorted(checkpoints, key=get_epoch)[-1]
        else:
            print(f"No checkpoints found in {args.base_dir}")
            sys.exit(1)
    else:
        # Default behavior: try latest
        checkpoints = sorted(
            glob.glob(os.path.join(args.base_dir, "checkpoint-epoch-*"))
        )
        if checkpoints:
            target_ckpt = sorted(checkpoints, key=get_epoch)[-1]
        else:
            parser.print_help()
            sys.exit(1)

    if not os.path.exists(target_ckpt):
        print(f"Checkpoint not found: {target_ckpt}")
        sys.exit(1)

    epoch_num = get_epoch(target_ckpt)

    # Check if samples for this epoch already exist
    # (Simple check: if sample_05_epochX.wav exists, skip?)
    # For now, we overwrite to be safe or if user re-runs

    success = generate(target_ckpt, epoch_num, args.output_dir)

    if success:
        update_index_html(args.output_dir)


if __name__ == "__main__":
    main()
