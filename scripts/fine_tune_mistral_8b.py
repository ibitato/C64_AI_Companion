#!/usr/bin/env python3
"""
Fine-tuning entrypoint for Ministral-3-8B-Thinking.

Supports:
  - DAPT (`data/processed/dapt/*.parquet`)
  - SFT (`data/processed/sft/*.jsonl`)
  - both (concatenated)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# AMD/ROCm compatibility hints
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCBLAS_TENSILE_LIBPATH"] = "/opt/rocm/lib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Ministral-3-8B-Thinking.")
    parser.add_argument("--mode", choices=["dapt", "sft", "both"], default="both")
    parser.add_argument("--model-path", default="models/Ministral-3-8B-Thinking")
    parser.add_argument("--dapt-dir", default="data/processed/dapt")
    parser.add_argument("--sft-dir", default="data/processed/sft")
    parser.add_argument("--output-dir", default="models/fine-tuned")

    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optim", default="adamw_torch")

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated list.",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def setup_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA/ROCm available: {device_count} device(s)")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  - {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1024**3:.2f} GB)")
        return "cuda", torch.float16
    print("CUDA/ROCm not available. Falling back to CPU.")
    return "cpu", torch.float32


def load_model_and_tokenizer(model_path: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        offload_folder="offload" if device == "cuda" else None,
        offload_state_dict=device == "cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _data_files(base_dir: str, extension: str) -> dict[str, str]:
    base = Path(base_dir)
    mapping = {}
    for split in ("train", "validation", "test"):
        p = base / f"{split}.{extension}"
        if p.exists():
            mapping[split] = str(p)
    return mapping


def load_dapt_dataset(dapt_dir: str) -> DatasetDict:
    data_files = _data_files(dapt_dir, "parquet")
    if not data_files:
        raise FileNotFoundError(f"No DAPT parquet files found in {dapt_dir}")
    return load_dataset("parquet", data_files=data_files)


def load_sft_dataset(sft_dir: str, tokenizer: AutoTokenizer) -> DatasetDict:
    data_files = _data_files(sft_dir, "jsonl")
    if not data_files:
        raise FileNotFoundError(f"No SFT jsonl files found in {sft_dir}")

    raw = load_dataset("json", data_files=data_files)

    def to_text(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    out = {}
    for split, ds in raw.items():
        out[split] = ds.map(to_text, remove_columns=ds.column_names)
    return DatasetDict(out)


def ensure_validation_split(ds: DatasetDict, seed: int) -> DatasetDict:
    if "validation" in ds:
        return ds
    split = ds["train"].train_test_split(test_size=0.05, seed=seed)
    return DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            **({"test": ds["test"]} if "test" in ds else {}),
        }
    )


def combine_modes(
    mode: str,
    dapt_ds: DatasetDict | None,
    sft_ds: DatasetDict | None,
) -> DatasetDict:
    if mode == "dapt":
        assert dapt_ds is not None
        return dapt_ds
    if mode == "sft":
        assert sft_ds is not None
        return sft_ds

    assert dapt_ds is not None and sft_ds is not None
    splits = sorted(set(dapt_ds.keys()) | set(sft_ds.keys()))
    out = {}
    for split in splits:
        if split in dapt_ds and split in sft_ds:
            out[split] = concatenate_datasets([dapt_ds[split], sft_ds[split]])
        elif split in dapt_ds:
            out[split] = dapt_ds[split]
        else:
            out[split] = sft_ds[split]
    return DatasetDict(out)


def tokenize_dataset(ds: DatasetDict, tokenizer: AutoTokenizer, max_length: int) -> DatasetDict:
    def tok(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
        enc["labels"] = [ids.copy() for ids in enc["input_ids"]]
        return enc

    out = {}
    for split, split_ds in ds.items():
        out[split] = split_ds.map(tok, batched=True, remove_columns=split_ds.column_names)
    return DatasetDict(out)


def configure_lora(model, args: argparse.Namespace):
    modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_training_args(args: argparse.Namespace, has_eval: bool) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        optim=args.optim,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        evaluation_strategy="steps" if has_eval else "no",
        load_best_model_at_end=has_eval,
    )


def main() -> None:
    args = parse_args()
    device, dtype = setup_device()

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=device, dtype=dtype)

    dapt_ds = None
    sft_ds = None

    if args.mode in ("dapt", "both"):
        print("Loading DAPT dataset...")
        dapt_ds = load_dapt_dataset(args.dapt_dir)

    if args.mode in ("sft", "both"):
        print("Loading SFT dataset...")
        sft_ds = load_sft_dataset(args.sft_dir, tokenizer)

    merged = combine_modes(args.mode, dapt_ds, sft_ds)
    merged = ensure_validation_split(merged, seed=args.seed)

    print("Tokenizing dataset...")
    tokenized = tokenize_dataset(merged, tokenizer, max_length=args.max_length)
    has_eval = "validation" in tokenized and len(tokenized["validation"]) > 0

    if args.use_lora:
        print("Applying LoRA adapters...")
        model = configure_lora(model, args)

    train_rows = len(tokenized["train"])
    val_rows = len(tokenized["validation"]) if has_eval else 0
    print(f"Training rows={train_rows} validation rows={val_rows}")

    train_args = build_training_args(args, has_eval=has_eval)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if has_eval else None,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Output saved in {args.output_dir}")


if __name__ == "__main__":
    main()
