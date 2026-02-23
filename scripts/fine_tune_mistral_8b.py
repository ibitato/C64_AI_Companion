#!/usr/bin/env python3
"""
Fine-tuning entrypoint for Ministral-3-8B-Thinking.

Phases:
  - dapt: domain adaptation on text chunks
  - sft: supervised chat fine-tuning with TRL SFTTrainer
  - both: run DAPT first, then SFT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Mistral3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

CANONICAL_MODEL_PATH = Path("models/Ministral-3-8B-Thinking").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Ministral-3-8B-Thinking.")
    parser.add_argument("--phase", choices=["dapt", "sft", "both"], default="both")
    # Backward-compatible alias.
    parser.add_argument("--mode", choices=["dapt", "sft", "both"], default=None, help=argparse.SUPPRESS)

    parser.add_argument("--model-path", default="models/Ministral-3-8B-Thinking")
    parser.add_argument("--dapt-dir", default="data/processed/dapt")
    parser.add_argument("--sft-dir", default="data/processed/sft")
    parser.add_argument("--output-dir", default="models/fine-tuned")
    parser.add_argument("--dapt-output-dir", default=None)
    parser.add_argument("--sft-output-dir", default=None)

    parser.add_argument("--precision", choices=["bf16", "fp16", "auto"], default="bf16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--eval-strategy", choices=["steps", "epoch", "no"], default="steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-scope",
        choices=["language_qkvo", "language_all_linear"],
        default="language_qkvo",
    )

    parser.add_argument("--assistant-only-loss", action="store_true", default=False)
    parser.add_argument("--no-assistant-only-loss", dest="assistant_only_loss", action="store_false")
    parser.add_argument("--packing", action="store_true", default=False)
    parser.add_argument("--no-packing", dest="packing", action="store_false")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback (debug only).")

    return parser.parse_args()


def normalize_phase(args: argparse.Namespace) -> None:
    if args.mode is not None:
        args.phase = args.mode


def enforce_model_policy(model_path: str) -> Path:
    resolved = Path(model_path).resolve()
    if resolved != CANONICAL_MODEL_PATH:
        raise ValueError(
            "Invalid base model path. This project only allows the original model at "
            f"{CANONICAL_MODEL_PATH} (received: {resolved})"
        )
    if not resolved.exists():
        raise FileNotFoundError(f"Model path not found: {resolved}")
    return resolved


def setup_device() -> tuple[str, bool]:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA/ROCm available: {device_count} device(s)")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  - {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1024**3:.2f} GB)")
        bf16_supported = torch.cuda.is_bf16_supported()
        print(f"bf16_supported={bf16_supported}")
        return "cuda", bf16_supported
    print("CUDA/ROCm not available. Falling back to CPU.")
    return "cpu", False


def resolve_dtype(precision: str, device: str, bf16_supported: bool) -> tuple[torch.dtype, bool, bool]:
    if device != "cuda":
        return torch.float32, False, False

    if precision == "bf16":
        if bf16_supported:
            return torch.bfloat16, True, False
        print("Requested bf16 but not supported. Falling back to fp16.")
        return torch.float16, False, True

    if precision == "fp16":
        return torch.float16, False, True

    # auto
    if bf16_supported:
        return torch.bfloat16, True, False
    return torch.float16, False, True


def load_base_model_and_tokenizer(model_path: str, device: str, dtype: torch.dtype):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "tie_word_embeddings"):
        config.tie_word_embeddings = False

    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if device == "cuda":
        model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _data_files(base_dir: str, extension: str) -> dict[str, str]:
    base = Path(base_dir)
    mapping: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        p = base / f"{split}.{extension}"
        if p.exists():
            mapping[split] = str(p)
    return mapping


def ensure_validation_split(ds: DatasetDict, seed: int) -> DatasetDict:
    if "validation" in ds and len(ds["validation"]) > 0:
        return ds
    split = ds["train"].train_test_split(test_size=0.05, seed=seed)
    out = DatasetDict({"train": split["train"], "validation": split["test"]})
    if "test" in ds:
        out["test"] = ds["test"]
    return out


def load_dapt_dataset(dapt_dir: str, seed: int) -> DatasetDict:
    data_files = _data_files(dapt_dir, "parquet")
    if not data_files:
        raise FileNotFoundError(f"No DAPT parquet files found in {dapt_dir}")
    ds = load_dataset("parquet", data_files=data_files)

    for split, split_ds in ds.items():
        if "text" not in split_ds.column_names:
            raise ValueError(f"DAPT split '{split}' must contain a 'text' column")

    return ensure_validation_split(ds, seed=seed)


def load_sft_dataset(sft_dir: str, seed: int) -> DatasetDict:
    data_files = _data_files(sft_dir, "jsonl")
    if not data_files:
        raise FileNotFoundError(f"No SFT jsonl files found in {sft_dir}")
    ds = load_dataset("json", data_files=data_files)

    for split, split_ds in ds.items():
        if "messages" not in split_ds.column_names:
            raise ValueError(f"SFT split '{split}' must contain a 'messages' column")

    return ensure_validation_split(ds, seed=seed)


def tokenize_dapt_dataset(ds: DatasetDict, tokenizer: AutoTokenizer, max_length: int) -> DatasetDict:
    def tok(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
        enc["labels"] = [ids.copy() for ids in enc["input_ids"]]
        return enc

    out: dict[str, Dataset] = {}
    for split, split_ds in ds.items():
        out[split] = split_ds.map(tok, batched=True, remove_columns=split_ds.column_names)
    return DatasetDict(out)


def _is_language_module(name: str) -> bool:
    return name.startswith("model.language_model.") or name.startswith("language_model.")


def find_lora_targets(model: torch.nn.Module, scope: str) -> list[str]:
    targets: list[str] = []

    if scope == "language_qkvo":
        suffixes = (".q_proj", ".k_proj", ".v_proj", ".o_proj")
        for name, _ in model.named_modules():
            if _is_language_module(name) and name.endswith(suffixes):
                targets.append(name)
    else:
        for name, module in model.named_modules():
            if not _is_language_module(name):
                continue
            if isinstance(module, torch.nn.Linear) and not name.endswith("lm_head"):
                targets.append(name)

    targets = sorted(set(targets))
    if not targets:
        raise ValueError(f"No LoRA targets found for scope '{scope}'")
    return targets


def apply_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    targets = find_lora_targets(model, args.lora_scope)
    print(f"LoRA target modules ({args.lora_scope}): {len(targets)}")
    print("Sample targets:", targets[:8])

    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def maybe_attach_adapter(model: torch.nn.Module, adapter_path: str | None) -> torch.nn.Module:
    if not adapter_path:
        return model
    adapter_file = Path(adapter_path) / "adapter_config.json"
    if not adapter_file.exists():
        return model
    print(f"Attaching existing LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    return model


def build_training_args(
    args: argparse.Namespace,
    output_dir: str,
    has_eval: bool,
    use_bf16: bool,
    use_fp16: bool,
) -> TrainingArguments:
    effective_eval = args.eval_strategy if has_eval else "no"
    load_best = effective_eval != "no" and has_eval

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        optim=args.optim,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        eval_strategy=effective_eval,
        load_best_model_at_end=load_best,
    )


def build_sft_args(
    args: argparse.Namespace,
    output_dir: str,
    has_eval: bool,
    use_bf16: bool,
    use_fp16: bool,
) -> SFTConfig:
    effective_eval = args.eval_strategy if has_eval else "no"
    load_best = effective_eval != "no" and has_eval

    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        optim=args.optim,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        eval_strategy=effective_eval,
        load_best_model_at_end=load_best,
        max_length=args.max_length,
        assistant_only_loss=args.assistant_only_loss,
        packing=args.packing,
    )


def run_dapt_phase(
    args: argparse.Namespace,
    model_path: str,
    output_dir: str,
    device: str,
    dtype: torch.dtype,
    use_bf16: bool,
    use_fp16: bool,
) -> tuple[str, str | None]:
    print("=== DAPT phase ===")
    model, tokenizer = load_base_model_and_tokenizer(model_path, device=device, dtype=dtype)

    if args.use_lora:
        model = apply_lora(model, args)

    dapt_ds = load_dapt_dataset(args.dapt_dir, seed=args.seed)
    tokenized = tokenize_dapt_dataset(dapt_ds, tokenizer, max_length=args.max_length)
    has_eval = "validation" in tokenized and len(tokenized["validation"]) > 0

    train_args = build_training_args(args, output_dir=output_dir, has_eval=has_eval, use_bf16=use_bf16, use_fp16=use_fp16)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print(f"DAPT rows train={len(tokenized['train'])} val={len(tokenized['validation']) if has_eval else 0}")
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if has_eval else None,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if args.use_lora:
        return model_path, output_dir
    return output_dir, None


def run_sft_phase(
    args: argparse.Namespace,
    model_path: str,
    output_dir: str,
    device: str,
    dtype: torch.dtype,
    use_bf16: bool,
    use_fp16: bool,
    adapter_path: str | None = None,
) -> tuple[str, str | None]:
    print("=== SFT phase ===")
    model, tokenizer = load_base_model_and_tokenizer(model_path, device=device, dtype=dtype)

    if args.assistant_only_loss:
        chat_template = tokenizer.chat_template or ""
        if "{% generation %}" not in chat_template:
            print(
                "assistant_only_loss requested, but tokenizer chat template does not expose "
                "{% generation %} mask blocks. Disabling assistant_only_loss for this run."
            )
            args.assistant_only_loss = False

    if adapter_path:
        model = maybe_attach_adapter(model, adapter_path)
    elif args.use_lora:
        model = apply_lora(model, args)

    sft_ds = load_sft_dataset(args.sft_dir, seed=args.seed)
    has_eval = "validation" in sft_ds and len(sft_ds["validation"]) > 0

    sft_args = build_sft_args(args, output_dir=output_dir, has_eval=has_eval, use_bf16=use_bf16, use_fp16=use_fp16)

    print(f"SFT rows train={len(sft_ds['train'])} val={len(sft_ds['validation']) if has_eval else 0}")
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=sft_ds["train"],
        eval_dataset=sft_ds["validation"] if has_eval else None,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if adapter_path or args.use_lora:
        return model_path, output_dir
    return output_dir, None


def main() -> None:
    args = parse_args()
    normalize_phase(args)
    canonical_model_path = enforce_model_policy(args.model_path)
    args.model_path = str(canonical_model_path)

    device, bf16_supported = setup_device()
    if device != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "GPU backend is not available. Training must run inside the ROCm 7.2 container. "
            "Use --allow-cpu only for debugging."
        )
    dtype, use_bf16, use_fp16 = resolve_dtype(args.precision, device=device, bf16_supported=bf16_supported)
    print(f"Effective precision: dtype={dtype}, bf16={use_bf16}, fp16={use_fp16}")

    if args.phase == "dapt":
        run_dapt_phase(
            args=args,
            model_path=args.model_path,
            output_dir=args.output_dir,
            device=device,
            dtype=dtype,
            use_bf16=use_bf16,
            use_fp16=use_fp16,
        )
        print(f"Done. DAPT output: {args.output_dir}")
        return

    if args.phase == "sft":
        run_sft_phase(
            args=args,
            model_path=args.model_path,
            output_dir=args.output_dir,
            device=device,
            dtype=dtype,
            use_bf16=use_bf16,
            use_fp16=use_fp16,
        )
        print(f"Done. SFT output: {args.output_dir}")
        return

    # both
    dapt_output = args.dapt_output_dir or f"{args.output_dir}-dapt"
    sft_output = args.sft_output_dir or args.output_dir

    next_model_path, adapter_path = run_dapt_phase(
        args=args,
        model_path=args.model_path,
        output_dir=dapt_output,
        device=device,
        dtype=dtype,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
    )

    run_sft_phase(
        args=args,
        model_path=next_model_path,
        output_dir=sft_output,
        device=device,
        dtype=dtype,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        adapter_path=adapter_path,
    )
    print(f"Done. DAPT output: {dapt_output}; SFT output: {sft_output}")


if __name__ == "__main__":
    main()
