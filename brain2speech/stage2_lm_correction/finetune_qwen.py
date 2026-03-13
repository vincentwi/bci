#!/usr/bin/env python3
"""
Stage 2c: Fine-tune Qwen3.5-2B with LoRA on phoneme correction task.

Uses DDP data parallelism (NOT pipeline/model parallelism) — the 2B model
fits on a single H100 in bf16 (~4GB), so each GPU gets a full copy and
processes its own batch. 6 GPUs = 6x throughput.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
        --num_processes 6 --mixed_precision bf16 \
        brain2speech/stage2_lm_correction/finetune_qwen.py

Outputs:
    brain2speech/models/qwen_phoneme_corrector/final/  — LoRA adapter weights
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODELS_DIR, QWEN_MODEL_NAME, SEED, SYSTEM_PROMPT


def analyze_embeddings(model, tokenizer, device="cuda"):
    """Pre-fine-tuning diagnostic: check if LM embeddings encode phoneme structure."""
    from sklearn.metrics.pairwise import cosine_similarity

    ARPABET_39 = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
    ]

    embeddings = {}
    base_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
    embed_layer = base_model.model.embed_tokens

    for phoneme in ARPABET_39:
        ids = tokenizer.encode(phoneme, add_special_tokens=False)
        with torch.no_grad():
            emb = embed_layer(torch.tensor(ids).to(device))
        embeddings[phoneme] = emb.mean(dim=0).float().cpu().numpy()

    emb_matrix = np.stack(list(embeddings.values()))
    sim = cosine_similarity(emb_matrix)

    noise_model_path = DATA_DIR / "noise_model.npy"
    if noise_model_path.exists():
        from scipy.stats import pearsonr
        from config import CLASS_TO_ARPABET, ARPABET_TO_CLASS
        C = np.load(noise_model_path)
        indices = [ARPABET_TO_CLASS[p] for p in ARPABET_39 if p in ARPABET_TO_CLASS]
        C_sub = C[np.ix_(indices, indices)]
        if C_sub.shape == sim.shape:
            r, p = pearsonr(sim.flatten(), C_sub.flatten())
            print(f"  Embedding-confusion correlation: r={r:.3f}, p={p:.2e}")

    np.save(DATA_DIR / "embedding_similarity.npy", sim)
    np.save(DATA_DIR / "embedding_phonemes.npy", np.array(ARPABET_39))
    print(f"  Saved embedding similarity matrix ({sim.shape})")
    return sim


class HealthCheckCallback:
    """Run a quick inference sanity check every N steps to verify the model
    is learning the correction task and not catastrophically forgetting."""

    TEST_CASES = [
        # (noisy, expected_clean_words)
        ("P AH T AH SIL W ER D", "butter word"),
        ("K AE SIL D AO G", "cat dog"),
        ("G UH D SIL M AO R N IH NG", "good morning"),
        ("F AH N IH SIL S AH N", "funny son"),
        ("S IY SIL DH AH SIL R EH D SIL B AO L", "see the red ball"),
    ]

    def __init__(self, tokenizer, check_every=100):
        self.tokenizer = tokenizer
        self.check_every = check_every
        self.history = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.check_every != 0 or state.global_step == 0:
            return
        # Only run on main process
        if state.is_world_process_zero:
            self._run_check(model, state.global_step)

    def _run_check(self, model, step):
        model.eval()
        results = []
        for noisy_input, expected_hint in self.TEST_CASES:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": noisy_input},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=60, do_sample=False,
                    num_beams=1, repetition_penalty=1.1,
                )
            response = self.tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            # Score: count how many phonemes match
            noisy_tokens = noisy_input.split()
            resp_tokens = response.split()
            valid_arpabet = {
                'AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER',
                'EY','F','G','HH','IH','IY','JH','K','L','M','N','NG','OW',
                'OY','P','R','S','SH','SIL','T','TH','UH','UW','V','W','Y','Z','ZH',
            }
            valid_out = [t for t in resp_tokens if t in valid_arpabet]
            is_valid = len(valid_out) > 0 and len(valid_out) >= len(noisy_tokens) * 0.5
            changed = sum(1 for a, b in zip(noisy_tokens, resp_tokens) if a != b)

            results.append({
                "input": noisy_input,
                "output": response[:80],
                "hint": expected_hint,
                "valid": is_valid,
                "changed": changed,
                "out_len": len(resp_tokens),
            })

        n_valid = sum(r["valid"] for r in results)
        avg_changed = np.mean([r["changed"] for r in results])

        print(f"\n{'='*60}")
        print(f"HEALTH CHECK @ step {step}")
        print(f"{'='*60}")
        print(f"  Valid outputs: {n_valid}/{len(results)}")
        print(f"  Avg phonemes changed: {avg_changed:.1f}")
        for r in results:
            status = "OK" if r["valid"] else "BAD"
            print(f"  [{status}] '{r['hint']}': {r['output'][:60]}")
        print(f"{'='*60}\n")

        self.history.append({
            "step": step,
            "n_valid": n_valid,
            "avg_changed": float(avg_changed),
            "results": results,
        })

        # Save history
        hist_path = DATA_DIR / "training_health_checks.json"
        with open(hist_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

        model.train()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    output_dir = str(MODELS_DIR / "qwen_phoneme_corrector")
    final_dir = str(MODELS_DIR / "qwen_phoneme_corrector" / "final")

    train_path = DATA_DIR / "phoneme_correction_train.jsonl"
    val_path = DATA_DIR / "phoneme_correction_val.jsonl"
    if not train_path.exists() or not val_path.exists():
        print("ERROR: Training data not found. Run generate_pairs.py first.")
        sys.exit(1)

    print("=" * 60)
    print("Stage 2c: Fine-tuning Qwen3.5-2B with LoRA (DDP)")
    print("=" * 60)

    # Load model — NO device_map="auto", let accelerate/DDP handle placement
    print(f"\nLoading {QWEN_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        # No device_map — DDP will replicate to each GPU
    )
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Embedding analysis (before LoRA, on CPU)
    print("\nAnalyzing pre-fine-tuning embeddings...")
    analyze_embeddings(model, tokenizer, device="cpu")

    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print("\nLoading training data...")
    dataset = load_dataset("json", data_files={
        "train": str(train_path),
        "validation": str(val_path),
    })
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Val: {len(dataset['validation'])} examples")

    # Training config — tuned for DDP on multiple H100s
    # With DDP, effective_batch = per_device_bs * grad_accum * n_processes
    # accelerate handles n_processes; we set per-device values
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,     # H100 has 80GB, model is ~4GB
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,       # No accumulation needed with large batch
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        max_length=128,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )

    # Health check callback — wrapping to match Trainer callback interface
    health_checker = HealthCheckCallback(tokenizer, check_every=100)

    class _HCCallback(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            health_checker.on_step_end(args, state, control, model=model, **kwargs)

    # Train
    print("\nStarting training...")
    t0 = time.time()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        callbacks=[_HCCallback()],
    )

    trainer.train()
    elapsed = time.time() - t0

    # Save final adapter
    print(f"\nSaving final adapter to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Final eval
    metrics = trainer.evaluate()
    print(f"\nFinal eval loss: {metrics['eval_loss']:.4f}")
    print(f"Total training time: {elapsed/60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
