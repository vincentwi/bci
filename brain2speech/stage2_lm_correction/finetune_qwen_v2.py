#!/usr/bin/env python3
"""
Stage 2c v2: Drastically improved Qwen3.5-2B LoRA fine-tuning.

Key changes over v1:
- LoRA r=64, alpha=128 — 4x more adapter capacity (~87M trainable params)
- Train on ALL projection layers + embed/lm_head for better phoneme encoding
- Regenerate data with 200k pairs, wider corruption rates, longer sequences
- Lower LR (5e-5) with warmup ratio 0.1 and cosine with restarts
- 8 epochs with early stopping (patience=5 eval steps)
- Gradient accumulation=4 for effective batch 768 across 6 GPUs
- Label smoothing 0.05 to reduce overconfidence
- Longer max_length=192 for more context

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
        --config_file configs/accelerate_ddp.yaml \
        --num_processes 6 \
        brain2speech/stage2_lm_correction/finetune_qwen_v2.py
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


# ── Step 0: Regenerate training data with better distribution ──

def _generate_chunk_v2(args):
    """Worker function for parallel v2 pair generation (module-level for pickling)."""
    from config import CLASS_TO_ARPABET, ARPABET_TO_CLASS

    chunk_id, n_pairs, words, word_phonemes, C, seed = args
    rng = np.random.RandomState(seed)
    pairs = []
    for _ in range(n_pairs):
        n_w = rng.randint(2, 9)
        chosen_idx = rng.choice(len(words), n_w, replace=False)

        clean = []
        chosen_names = []
        for wi in chosen_idx:
            clean.extend(word_phonemes[wi])
            clean.append('SIL')
            chosen_names.append(words[wi])
        clean = clean[:-1]

        # Bimodal corruption: 70% realistic (0.05-0.25), 30% hard (0.25-0.60)
        if rng.random() < 0.7:
            rate = rng.uniform(0.05, 0.25)
        else:
            rate = rng.uniform(0.25, 0.60)

        noisy = []
        for p in clean:
            if p == 'SIL' or p not in ARPABET_TO_CLASS:
                noisy.append(p)
                continue
            idx = ARPABET_TO_CLASS[p]
            if rng.random() < rate:
                noisy_idx = rng.choice(len(C), p=C[idx])
                noisy.append(CLASS_TO_ARPABET[noisy_idx])
            else:
                noisy.append(p)

        # Insertion/deletion noise (5% chance each)
        if rng.random() < 0.05 and len(noisy) > 3:
            non_sil = [i for i, p in enumerate(noisy) if p != 'SIL']
            if non_sil:
                del_idx = rng.choice(non_sil)
                noisy.pop(del_idx)
                clean.pop(del_idx)

        if rng.random() < 0.05:
            insert_idx = rng.randint(0, len(noisy))
            random_ph = CLASS_TO_ARPABET[rng.randint(0, len(CLASS_TO_ARPABET))]
            noisy.insert(insert_idx, random_ph)
            clean.insert(insert_idx, clean[min(insert_idx, len(clean)-1)])

        pairs.append({
            "noisy": " ".join(noisy),
            "clean": " ".join(clean),
            "words": " ".join(chosen_names),
            "corruption_rate": float(rate),
            "n_phonemes": len(clean),
        })
    return pairs


def _format_as_chat_v2(pair):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["noisy"]},
            {"role": "assistant", "content": pair["clean"]},
        ]
    }


def regenerate_training_data():
    """Generate 200k pairs with improved corruption distribution."""
    from config import CLASS_TO_ARPABET, ARPABET_TO_CLASS
    from multiprocessing import Pool, cpu_count

    noise_model_path = DATA_DIR / "noise_model.npy"
    cmudict_path = DATA_DIR / "cmudict-0.7b"

    if not noise_model_path.exists() or not cmudict_path.exists():
        print("ERROR: noise_model.npy or cmudict not found")
        sys.exit(1)

    C = np.load(noise_model_path)

    entries = {}
    with open(cmudict_path, 'r', encoding='latin-1') as f:
        for line in f:
            if line.startswith(';;;') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].split('(')[0]
            phonemes = [p.rstrip('012') for p in parts[1:]]
            if word not in entries:
                entries[word] = phonemes

    valid_phonemes = set(CLASS_TO_ARPABET.values()) - {'SIL'}
    cmudict = {w: p for w, p in entries.items() if all(ph in valid_phonemes for ph in p)}
    print(f"  CMU dict: {len(cmudict)} entries")

    words = list(cmudict.keys())
    word_phonemes = [cmudict[w] for w in words]

    n_pairs = 200000
    n_workers = min(cpu_count(), 16)
    chunk_size = n_pairs // n_workers
    remainder = n_pairs % n_workers

    args_list = []
    for i in range(n_workers):
        n = chunk_size + (1 if i < remainder else 0)
        args_list.append((i, n, words, word_phonemes, C, SEED + 1000 + i))

    print(f"  Generating {n_pairs} v2 pairs across {n_workers} workers...")
    with Pool(n_workers) as pool:
        chunks = pool.map(_generate_chunk_v2, args_list)

    pairs = []
    for chunk in chunks:
        pairs.extend(chunk)

    np.random.seed(SEED + 2000)
    np.random.shuffle(pairs)

    train_pairs = pairs[:160000]
    val_pairs = pairs[160000:180000]
    test_pairs = pairs[180000:]

    for name, data in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        path = DATA_DIR / f"phoneme_correction_{name}_v2.jsonl"
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(_format_as_chat_v2(item)) + '\n')
        print(f"  Saved {len(data)} {name} examples to {path}")

    raw_path = DATA_DIR / "phoneme_correction_test_raw_v2.jsonl"
    with open(raw_path, 'w') as f:
        for item in test_pairs:
            f.write(json.dumps(item) + '\n')
    print(f"  Saved {len(test_pairs)} raw test pairs")

    avg_len = np.mean([p["n_phonemes"] for p in pairs])
    avg_rate = np.mean([p["corruption_rate"] for p in pairs])
    print(f"  Stats: avg length={avg_len:.1f}, avg corruption={avg_rate:.2f}")


class HealthCheckCallback:
    """Run inference sanity check every N steps."""

    TEST_CASES = [
        ("P AH T AH SIL W ER D", "butter word"),
        ("K AE SIL D AO G", "cat dog"),
        ("G UH D SIL M AO R N IH NG", "good morning"),
        ("F AH N IH SIL S AH N", "funny son"),
        ("S IY SIL DH AH SIL R EH D SIL B AO L", "see the red ball"),
        ("T AE SIL K AH M SIL HH IY R", "take come here"),
        ("B IH G SIL R EH D SIL K AA R", "big red car"),
    ]

    def __init__(self, tokenizer, check_every=100):
        self.tokenizer = tokenizer
        self.check_every = check_every
        self.history = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.check_every != 0 or state.global_step == 0:
            return
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
                "input": noisy_input, "output": response[:80],
                "hint": expected_hint, "valid": is_valid,
                "changed": changed, "out_len": len(resp_tokens),
            })

        n_valid = sum(r["valid"] for r in results)
        avg_changed = np.mean([r["changed"] for r in results])

        print(f"\n{'='*60}")
        print(f"HEALTH CHECK v2 @ step {step}")
        print(f"{'='*60}")
        print(f"  Valid outputs: {n_valid}/{len(results)}")
        print(f"  Avg phonemes changed: {avg_changed:.1f}")
        for r in results:
            status = "OK" if r["valid"] else "BAD"
            print(f"  [{status}] '{r['hint']}': {r['output'][:60]}")
        print(f"{'='*60}\n")

        self.history.append({
            "step": step, "n_valid": n_valid,
            "avg_changed": float(avg_changed), "results": results,
        })
        with open(DATA_DIR / "training_health_checks_v2.json", 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

        model.train()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, EarlyStoppingCallback
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    output_dir = str(MODELS_DIR / "qwen_phoneme_corrector_v2")
    final_dir = str(MODELS_DIR / "qwen_phoneme_corrector_v2" / "final")

    # Step 0: Regenerate data with improved distribution
    v2_train = DATA_DIR / "phoneme_correction_train_v2.jsonl"
    v2_val = DATA_DIR / "phoneme_correction_val_v2.jsonl"
    if not v2_train.exists() or not v2_val.exists():
        print("Regenerating v2 training data...")
        regenerate_training_data()

    print("=" * 60)
    print("Stage 2c v2: Drastically improved LoRA fine-tuning")
    print("=" * 60)

    # Load model from scratch — NO resume from v1
    print(f"\nLoading {QWEN_MODEL_NAME} (fresh start)...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    # LoRA v2 — 2x capacity vs v1, moderate dropout
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.08,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load v2 dataset
    print("\nLoading v2 training data...")
    dataset = load_dataset("json", data_files={
        "train": str(v2_train),
        "validation": str(v2_val),
    })
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Val: {len(dataset['validation'])} examples")

    # v2 training config — key changes: 2x data, grad_accum=2, 5 epochs, early stop
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,         # effective_bs = 32*2*6 = 384
        learning_rate=1.5e-4,                  # Slightly lower than v1's 2e-4
        lr_scheduler_type="cosine",
        warmup_steps=100,                      # Fixed warmup, not ratio
        weight_decay=0.01,
        bf16=True,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=150,                        # Eval frequently for early stopping
        save_strategy="steps",
        save_steps=150,
        save_total_limit=5,
        max_length=192,                        # Longer context for longer sequences
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )

    health_checker = HealthCheckCallback(tokenizer, check_every=100)

    class _HCCallback(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            health_checker.on_step_end(args, state, control, model=model, **kwargs)

    # Early stopping — stop if eval_loss doesn't improve for 5 evals (500 steps)
    early_stop = EarlyStoppingCallback(early_stopping_patience=5)

    print("\nStarting v2 training (early stopping enabled)...")
    t0 = time.time()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        callbacks=[_HCCallback(), early_stop],
    )

    trainer.train()
    elapsed = time.time() - t0

    print(f"\nSaving final adapter to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    metrics = trainer.evaluate()
    print(f"\nFinal eval loss: {metrics['eval_loss']:.4f}")
    print(f"Total training time: {elapsed/60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
