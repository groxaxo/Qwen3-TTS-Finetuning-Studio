#!/usr/bin/env python3
"""Generate samples from ALL checkpoints for side-by-side comparison."""
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import os
import json
from datetime import datetime
import gc

CKPT_BASE = "output_model_monica_17b_v2"
OUTPUT_DIR = "comparison_samples"
EPOCHS = list(range(20))  # 0-19

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test phrases for comparison
TEST_PHRASES = [
    {
        "id": 1,
        "text": "Hola, soy Monica. Bienvenidos a mi demo de síntesis de voz.",
        "category": "Greeting"
    },
    {
        "id": 2,
        "text": "Tres tristes tigres tragaban trigo en un trigal.",
        "category": "Tongue Twister"
    },
    {
        "id": 3,
        "text": "Buenos Aires es una ciudad increíble, llena de cultura y pasión.",
        "category": "Culture"
    },
]

results = {
    "generated_at": datetime.now().isoformat(),
    "checkpoint_base": CKPT_BASE,
    "epochs": EPOCHS,
    "phrases": TEST_PHRASES,
    "samples": {}
}

print("=" * 70)
print("Checkpoint Comparison Sample Generator")
print("=" * 70)
print(f"Checkpoints: {CKPT_BASE}/checkpoint-epoch-{{0-19}}")
print(f"Phrases: {len(TEST_PHRASES)}")
print(f"Total samples: {len(EPOCHS) * len(TEST_PHRASES)}")
print("=" * 70)

for epoch in EPOCHS:
    ckpt_path = f"{CKPT_BASE}/checkpoint-epoch-{epoch}"
    print(f"\n{'='*70}")
    print(f"Loading Epoch {epoch}: {ckpt_path}")
    print("=" * 70)
    
    # Create epoch directory
    epoch_dir = f"{OUTPUT_DIR}/epoch-{epoch:02d}"
    os.makedirs(epoch_dir, exist_ok=True)
    
    try:
        tts = Qwen3TTSModel.from_pretrained(
            ckpt_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        
        spk = tts.get_supported_speakers()[0]
        epoch_results = []
        
        for phrase in TEST_PHRASES:
            print(f"  [{phrase['id']}/{len(TEST_PHRASES)}] {phrase['text'][:50]}...")
            
            wavs, sr = tts.generate_custom_voice(
                text=phrase["text"],
                language="Spanish",
                speaker=spk,
                max_new_tokens=512,
            )
            
            filename = f"phrase_{phrase['id']:02d}.wav"
            filepath = f"{epoch_dir}/{filename}"
            sf.write(filepath, wavs[0], sr)
            
            duration = len(wavs[0]) / sr
            print(f"       ✓ Duration: {duration:.2f}s")
            
            epoch_results.append({
                "phrase_id": phrase["id"],
                "filename": filename,
                "duration": round(duration, 2),
            })
        
        results["samples"][f"epoch-{epoch}"] = {
            "checkpoint": ckpt_path,
            "speaker": spk,
            "sample_rate": sr,
            "results": epoch_results
        }
        
        # Clean up GPU memory
        del tts
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results["samples"][f"epoch-{epoch}"] = {"error": str(e)}

# Save metadata
with open(f"{OUTPUT_DIR}/metadata.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print(f"✓ Generated samples for {len(EPOCHS)} epochs")
print(f"✓ Metadata saved to {OUTPUT_DIR}/metadata.json")
print("=" * 70)
