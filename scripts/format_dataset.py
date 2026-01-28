
import os
import argparse
import librosa
import soundfile as sf
import json
from tqdm import tqdm
from pathlib import Path
import shutil

def process_dataset(dataset_dir, output_dir, metadata_file=None, ref_audio_path=None):
    """
    Process a dataset into Qwen3-TTS compatible JSONL format and 24kHz audio.
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_audio_dir = output_path / "wavs"
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find metadata file if not specified
    if metadata_file is None:
        possible_metadata = list(dataset_path.glob("metadata.csv")) + list(dataset_path.glob("metadata.txt"))
        if not possible_metadata:
            raise FileNotFoundError(f"No metadata.csv or metadata.txt found in {dataset_dir}. Please specify --metadata_file.")
        metadata_file = possible_metadata[0]
    
    print(f"Using metadata file: {metadata_file}")

    # 2. Prepare Reference Audio
    final_ref_audio_path = None
    if ref_audio_path:
        ref_src = Path(ref_audio_path)
        if not ref_src.exists():
             raise FileNotFoundError(f"Reference audio not found at {ref_audio_path}")
        
        # Resample ref audio too just in case
        ref_y, sr = librosa.load(ref_src, sr=24000, mono=True)
        final_ref_audio_path = output_path / "ref_audio.wav"
        sf.write(str(final_ref_audio_path), ref_y, 24000)
        print(f"Processed reference audio: {final_ref_audio_path}")
    
    # 3. Process lines
    jsonl_lines = []
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Found {len(lines)} utterances. Processing...")

    first_audio_processed = None

    for line in tqdm(lines):
        # Support LJSpeech format: filename|transcript|normalized_transcript(optional)
        parts = line.strip().split('|')
        if len(parts) < 2:
            continue
        
        filename = parts[0]
        text = parts[1]
        
        # Locate audio file - try adding .wav if missing
        audio_src = dataset_path / "wavs" / filename
        if not audio_src.exists():
            audio_src = dataset_path / "wavs" / f"{filename}.wav"
        
        if not audio_src.exists():
            # Try looking in root if not in wavs/
            audio_src = dataset_path / filename
            if not audio_src.exists():
                audio_src = dataset_path / f"{filename}.wav"
        
        if not audio_src.exists():
            print(f"Warning: Audio file for {filename} not found, skipping.")
            continue

        # Load and Resample
        try:
            y, sr = librosa.load(audio_src, sr=24000, mono=True)
        except Exception as e:
            print(f"Error reading {audio_src}: {e}")
            continue

        # Save to output
        out_filename = f"{Path(filename).stem}.wav"
        out_file = output_audio_dir / out_filename
        sf.write(str(out_file), y, 24000)

        # Handle Reference Audio (if not provided externally, use the first processed file)
        if final_ref_audio_path is None:
            if first_audio_processed is None:
                first_audio_processed = out_file
                # Copy this as the ref audio for consistency
                final_ref_audio_path = output_path / "ref_audio_auto.wav"
                shutil.copy(out_file, final_ref_audio_path)
                print(f"No reference audio provided. Selected {filename} as reference.")

        entry = {
            "audio": str(out_file.absolute()),
            "text": text,
            "ref_audio": str(final_ref_audio_path.absolute())
        }
        jsonl_lines.append(json.dumps(entry, ensure_ascii=False))

    # 4. Write JSONL
    out_jsonl = output_path / "train_raw.jsonl"
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        f.write('\n'.join(jsonl_lines))
    
    print(f"Done! Saved {len(jsonl_lines)} samples to {out_jsonl}")
    print(f"You can now run prepare_data.py using this file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format dataset for Qwen3-TTS finetuning")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to source dataset directory (should contain metadata.csv and wavs/)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--metadata_file", type=str, help="Path to metadata file (optional, defaults to metadata.csv inside dataset_dir)")
    parser.add_argument("--ref_audio", type=str, help="Path to a reference audio file to be used for all samples (recommended)")
    
    args = parser.parse_args()
    
    process_dataset(args.dataset_dir, args.output_dir, args.metadata_file, args.ref_audio)
