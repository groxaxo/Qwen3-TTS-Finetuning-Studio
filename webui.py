
import gradio as gr
import os
import sys
import subprocess
import torch
import soundfile as sf
import threading
from qwen_tts import Qwen3TTSModel
from pathlib import Path

# Add scripts to path to allow importing if needed, though we might use subprocess for isolation
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

def run_format_dataset(dataset_dir, output_dir, ref_audio):
    cmd = [
        sys.executable, "scripts/format_dataset.py",
        "--dataset_dir", dataset_dir,
        "--output_dir", output_dir
    ]
    if ref_audio:
        cmd.extend(["--ref_audio", ref_audio])
    
    yield f"Running command: {' '.join(cmd)}\n"
    
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        yield line
    p.wait()
    if p.returncode == 0:
        yield "\nDataset formatting completed successfully!"
    else:
        yield f"\nError: Process exited with code {p.returncode}"

def run_prepare_data(input_jsonl, output_jsonl, device, tokenizer_path):
    cmd = [
        sys.executable, "prepare_data.py",
        "--device", device,
        "--tokenizer_model_path", tokenizer_path,
        "--input_jsonl", input_jsonl,
        "--output_jsonl", output_jsonl
    ]
    
    yield f"Running command: {' '.join(cmd)}\n"
    
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        yield line
    p.wait()
    if p.returncode == 0:
        yield "\nData preparation completed successfully!"
    else:
        yield f"\nError: Process exited with code {p.returncode}"

def run_finetuning(init_model, output_path, train_jsonl, batch_size, lr, epochs, speaker_name):
    cmd = [
        sys.executable, "sft_12hz.py",
        "--init_model_path", init_model,
        "--output_model_path", output_path,
        "--train_jsonl", train_jsonl,
        "--batch_size", str(int(batch_size)),
        "--lr", str(lr),
        "--num_epochs", str(int(epochs)),
        "--speaker_name", speaker_name
    ]
    
    yield f"Running command: {' '.join(cmd)}\n"
    
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        yield line
    p.wait()
    if p.returncode == 0:
        yield "\nTraining completed successfully!"
    else:
        yield f"\nError: Process exited with code {p.returncode}"

# Inference Cache
model_cache = {}

def load_and_generate(checkpoint_path, text, speaker_name, device):
    global model_cache
    
    try:
        if checkpoint_path not in model_cache:
            yield "Loading model... (this may take a while)\n", None
            model = Qwen3TTSModel.from_pretrained(
                checkpoint_path,
                device_map=device,
                dtype=torch.bfloat16,
#                attn_implementation="flash_attention_2", # verify if user has this supported, usually safer to defaults or auto
            )
            model_cache[checkpoint_path] = model
        else:
            model = model_cache[checkpoint_path]
        
        yield "Generating audio...\n", None
        
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=speaker_name,
        )
        
        output_file = "output_generated.wav"
        sf.write(output_file, wavs[0], sr)
        
        yield "Generation complete!", output_file
        
    except Exception as e:
        yield f"Error: {str(e)}", None

with gr.Blocks(title="Qwen3-TTS Finetuning Studio") as demo:
    gr.Markdown("# 🎙️ Qwen3-TTS Finetuning Studio")
    
    with gr.Tab("1. Format Dataset"):
        gr.Markdown("Step 1: Convert your raw folder of wavs into the format Qwen needs (24kHz re-sampling + jsonl generation).")
        with gr.Row():
            ds_dir = gr.Textbox(label="Raw Dataset Directory", placeholder="/path/to/my_wavs_and_metadata")
            ds_out = gr.Textbox(label="Output Directory", placeholder="./my_formatted_data")
        ds_ref = gr.Textbox(label="Reference Audio Path (Optional - will pick first one otherwise)")
        ds_btn = gr.Button("Format Dataset", variant="primary")
        ds_log = gr.Textbox(label="Logs", lines=10)
        
        ds_btn.click(run_format_dataset, [ds_dir, ds_out, ds_ref], ds_log)

    with gr.Tab("2. Prepare Codes"):
        gr.Markdown("Step 2: Extract audio codes from the formatted JSONL.")
        with gr.Row():
            prep_in = gr.Textbox(label="Input JSONL (from Step 1)", value="./my_formatted_data/train_raw.jsonl")
            prep_out = gr.Textbox(label="Output JSONL", value="./my_formatted_data/train_with_codes.jsonl")
        
        with gr.Row():
            prep_dev = gr.Textbox(label="Device", value="cuda:0")
            prep_tok = gr.Textbox(label="Tokenizer Path", value="Qwen/Qwen3-TTS-Tokenizer-12Hz")
            
        prep_btn = gr.Button("Prepare Data", variant="primary")
        prep_log = gr.Textbox(label="Logs", lines=10)
        
        prep_btn.click(run_prepare_data, [prep_in, prep_out, prep_dev, prep_tok], prep_log)

    with gr.Tab("3. Fine-Tune"):
        gr.Markdown("Step 3: Run the training loop.")
        with gr.Row():
            ft_train_jsonl = gr.Textbox(label="Training JSONL (from Step 2)", value="./my_formatted_data/train_with_codes.jsonl")
            ft_out_dir = gr.Textbox(label="Output Model Directory", value="./output_model")
        
        with gr.Row():
            ft_base_model = gr.Textbox(label="Base Model", value="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
            ft_speaker = gr.Textbox(label="Speaker Name", value="my_custom_voice")
        
        with gr.Row():
            ft_bs = gr.Number(label="Batch Size", value=2)
            ft_epochs = gr.Number(label="Epochs", value=10)
            ft_lr = gr.Textbox(label="Learning Rate", value="2e-5")
            
        ft_btn = gr.Button("Start Training", variant="primary")
        ft_log = gr.Textbox(label="Logs", lines=10)
        
        ft_btn.click(run_finetuning, [ft_base_model, ft_out_dir, ft_train_jsonl, ft_bs, ft_lr, ft_epochs, ft_speaker], ft_log)
        
    with gr.Tab("4. Inference"):
        gr.Markdown("Test your fine-tuned model.")
        with gr.Row():
            inf_ckpt = gr.Textbox(label="Checkpoint Path", placeholder="./output_model/checkpoint-epoch-X")
            inf_spk = gr.Textbox(label="Speaker Name", placeholder="my_custom_voice")
            inf_dev = gr.Textbox(label="Device", value="cuda:0")
        
        inf_text = gr.TextArea(label="Text to Speak", value="Hello, this is my cloned voice.")
        inf_btn = gr.Button("Generate", variant="primary")
        
        with gr.Row():
            inf_status = gr.Textbox(label="Status")
            inf_audio = gr.Audio(label="Output Audio")
            
        inf_btn.click(load_and_generate, [inf_ckpt, inf_text, inf_spk, inf_dev], [inf_status, inf_audio])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
