#!/usr/bin/env python3
"""
Watch output directory for new checkpoints and auto-generate Spanish samples.
"""

import time
import os
import subprocess
import glob

# Configuration
CKPT_BASE = "output_model_monica_17b_v4"
OUTPUT_DIR = "spanish_samples"
POLL_INTERVAL = 30  # seconds


def get_epoch(path):
    try:
        return int(path.split("-")[-1])
    except ValueError:
        return -1


def get_checkpoints():
    return sorted(glob.glob(os.path.join(CKPT_BASE, "checkpoint-epoch-*")))


def is_checkpoint_ready(path):
    # Check if model.safetensors exists and size is stable (or just exists)
    return os.path.exists(os.path.join(path, "model.safetensors"))


def has_generated_for_epoch(epoch):
    # Check if samples exist for this epoch
    # We check for sample_05_epoch{epoch}.wav as a proxy
    marker = os.path.join(OUTPUT_DIR, f"sample_05_epoch{epoch}.wav")
    return os.path.exists(marker)


def main():
    print(f"👀 Watching {CKPT_BASE} for new checkpoints...")
    print(f"   Outputs will go to {OUTPUT_DIR}")
    print("   Press Ctrl+C to stop.")

    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # We use the conda python that has the deps
    python_cmd = "/home/op/miniconda/envs/qwen3tts17b/bin/python"
    script_path = "generate_spanish_samples.py"

    while True:
        try:
            ckpts = get_checkpoints()
            for ckpt in ckpts:
                epoch = get_epoch(ckpt)

                if epoch < 0:
                    continue

                if not is_checkpoint_ready(ckpt):
                    continue

                if has_generated_for_epoch(epoch):
                    continue

                print(f"\n[NEW] Found new checkpoint: Epoch {epoch}")
                print(f"      Triggering generation...")

                # Run generation
                subprocess.run(
                    [
                        python_cmd,
                        script_path,
                        "--checkpoint",
                        ckpt,
                        "--output_dir",
                        OUTPUT_DIR,
                    ],
                    check=False,
                )

                print(f"      Done processing Epoch {epoch}\n")

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\nStopping watcher.")
            break
        except Exception as e:
            print(f"Error in watcher: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
