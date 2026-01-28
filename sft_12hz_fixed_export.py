# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

from dataset import TTSDataset

# Prefer official import surface
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def resolve_model_dir(model_id_or_path: str) -> str:
    """
    Resolve model directory from path or HuggingFace model ID.
    
    Args:
        model_id_or_path: Local directory path or HuggingFace model ID
        
    Returns:
        Absolute path to the model directory
    """
    if os.path.isdir(model_id_or_path):
        return os.path.abspath(model_id_or_path)
    # download (or reuse cache)
    return os.path.abspath(snapshot_download(repo_id=model_id_or_path, repo_type="model"))


def load_jsonl(path: str):
    """
    Load data from JSONL file.
    
    Args:
        path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_last_hidden(outputs):
    """
    Extract the last hidden state from model outputs.
    
    Handles different nesting structures in Qwen3-TTS model outputs.
    
    Args:
        outputs: Model output object with hidden_states attribute
        
    Returns:
        Last hidden state tensor
        
    Raises:
        RuntimeError: If hidden_states is not present in outputs
    """
    hs = getattr(outputs, "hidden_states", None)
    if hs is None:
        raise RuntimeError("Model did not return hidden_states. Ensure output_hidden_states=True.")
    # Some Qwen3-TTS wrappers may nest; handle both cases.
    if isinstance(hs, (list, tuple)) and len(hs) > 0 and isinstance(hs[0], (list, tuple)):
        return hs[0][-1]
    if isinstance(hs, (list, tuple)):
        return hs[-1]
    return hs


def train():
    """
    Main training function for Qwen3-TTS fine-tuning.
    
    This script fixes three critical bugs from the original sft_12hz.py:
    1. Text projection only applied when dimensions mismatch (fixes 1.7B gibberish)
    2. Sub-talker alignment properly shifts both mask and targets
    3. Speaker embedding uses running mean across all batches (not just first)
    
    Supports two export modes:
    - custom_voice: Bakes speaker embedding into model (simple inference)
    - base: Keeps speaker encoder for flexible voice cloning
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--output_model_path", type=str, default="output")
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--num_epochs", type=int, default=3)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--speaker_name", type=str, required=True)

    # export mode:
    #   - custom_voice: bakes a fixed speaker id (generate_custom_voice)
    #   - base: keeps speaker_encoder and base-style config (generate_voice_clone)
    ap.add_argument("--export_mode", type=str, choices=["custom_voice", "base"], default="custom_voice")
    ap.add_argument("--speaker_index", type=int, default=3000)  # only used for custom_voice export
    ap.add_argument("--attn_impl", type=str, default="sdpa")    # safer than flash_attention_2
    args = ap.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
        log_with="tensorboard",
    )

    base_dir = resolve_model_dir(args.init_model_path)

    # Load model + processor
    tts = Qwen3TTSModel.from_pretrained(
        base_dir,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    config = AutoConfig.from_pretrained(base_dir)

    train_rows = load_jsonl(args.train_jsonl)
    dataset = TTSDataset(train_rows, tts.processor, config)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, dl = accelerator.prepare(tts.model, optimizer, dl)
    model.train()

    # Running mean speaker embedding (CPU fp32), only used for custom_voice export
    spk_mean = None
    spk_n = 0

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(dl):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]              # [B,S,2]
                codec_ids = batch["codec_ids"]              # [B,S,16]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]            # [B,S] bool

                # device/dtype
                p = next(model.parameters())
                dev = p.device
                dt = p.dtype

                # Speaker embedding from ref audio
                speaker_embedding = model.speaker_encoder(ref_mels.to(dev).to(dt)).detach()  # [B,H]

                # Update running mean (for custom_voice export)
                emb = speaker_embedding.float().mean(dim=0, keepdim=True).cpu()
                if spk_mean is None:
                    spk_mean = emb.clone()
                    spk_n = 1
                else:
                    spk_n += 1
                    spk_mean += (emb - spk_mean) / spk_n

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # Raw text embedding (official behavior)
                text_emb = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask

                # Codec embedding
                codec_emb = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask

                # ✅ ONLY project for 0.6B (dim mismatch), never blindly
                if text_emb.shape[-1] != codec_emb.shape[-1]:
                    if not hasattr(model.talker, "text_projection"):
                        raise RuntimeError("Dim mismatch but model has no text_projection.")
                    text_emb = model.talker.text_projection(text_emb)

                # Inject speaker embedding at slot 6 (special position for speaker info)
                SPEAKER_EMBEDDING_SLOT = 6
                codec_emb[:, SPEAKER_EMBEDDING_SLOT, :] = speaker_embedding

                inputs_embeds = text_emb + codec_emb

                # Add codebook 1..15 embeddings (codebook 0 is handled separately via labels)
                NUM_SUB_CODEBOOKS = 16
                for i in range(1, NUM_SUB_CODEBOOKS):
                    emb_i = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    emb_i = emb_i * codec_mask.unsqueeze(-1)
                    inputs_embeds = inputs_embeds + emb_i

                outputs = model.talker(
                    inputs_embeds=inputs_embeds[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = get_last_hidden(outputs)  # [B,S-1,H]

                # ✅ CRITICAL: align mask + targets (shift BOTH)
                mask = codec_mask[:, 1:]
                talker_hidden_states = hidden_states[mask]        # [N,H]
                talker_codec_ids = codec_ids[:, 1:, :][mask]      # [N,16]

                _, sub_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + sub_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Log progress every N steps
            LOG_INTERVAL = 10
            if step % LOG_INTERVAL == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        # ---- Save checkpoint per epoch ----
        if accelerator.is_main_process:
            out_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            os.makedirs(out_dir, exist_ok=True)

            # Copy base artifacts (tokenizer, configs, etc.)
            shutil.copytree(base_dir, out_dir, dirs_exist_ok=True)

            # Patch config depending on export mode
            cfg_path = os.path.join(out_dir, "config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            if args.export_mode == "custom_voice":
                cfg["tts_model_type"] = "custom_voice"
                spk_id = {args.speaker_name: int(args.speaker_index)}
                spk_dialect = {args.speaker_name: False}

                # Put in BOTH locations for compatibility
                cfg["spk_id"] = spk_id
                cfg["spk_is_dialect"] = spk_dialect
                tc = cfg.get("talker_config", {})
                tc["spk_id"] = spk_id
                tc["spk_is_dialect"] = spk_dialect
                cfg["talker_config"] = tc

            else:
                # Base-style export: keep it as Base (do NOT force custom_voice)
                # Ensure we do NOT add spk_id mapping
                cfg.pop("spk_id", None)
                cfg.pop("spk_is_dialect", None)
                tc = cfg.get("talker_config", {})
                tc.pop("spk_id", None)
                tc.pop("spk_is_dialect", None)
                cfg["talker_config"] = tc

            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

            # Save weights
            unwrapped = accelerator.unwrap_model(model)
            state = {k: v.detach().cpu() for k, v in unwrapped.state_dict().items()}

            if args.export_mode == "custom_voice":
                # Drop speaker encoder to shrink + because we bake the embedding
                for k in list(state.keys()):
                    if k.startswith("speaker_encoder"):
                        del state[k]

                # Bake speaker embedding into codec embedding table
                w = state["talker.model.codec_embedding.weight"]
                idx = int(args.speaker_index)
                if idx < 0 or idx >= w.shape[0]:
                    raise RuntimeError(f"speaker_index {idx} out of range (vocab={w.shape[0]})")
                if spk_mean is None:
                    raise RuntimeError("spk_mean is None; no batches seen?")
                w[idx] = spk_mean[0].to(w.dtype)

            # Base export keeps speaker_encoder intact (so generate_voice_clone still works)

            save_file(state, os.path.join(out_dir, "model.safetensors"))
            accelerator.print(f"[SAVED] {out_dir}  (export_mode={args.export_mode})")


if __name__ == "__main__":
    train()
