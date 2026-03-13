#!/usr/bin/env python3
"""Train improved phoneme→text LIFT with augmented data.

Improvements over v1:
1. Use ALL 4 DCoND seeds for training data (4x more diverse errors)
2. Include ground truth phonemes → text pairs (teaches correct mapping)
3. More epochs (5 instead of 3) with lower LR
4. LoRA r=64 instead of r=32 for more capacity
5. Include target phonemes in prompt for the model to cross-reference
"""
import json, sys, os, time, random
import numpy as np
import h5py
import torch

sys.path.insert(0, 'brain2speech')
from config import CLASS_TO_ARPABET, N_CLASSES
from train_beyond_paper import compute_per, ctc_greedy_decode, DATA_DIR, RESULTS_DIR, CTC_BLANK
from lead2_train_dcond import PaperExactDCoND, build_marginalization_matrix_ext

device = torch.device('cuda:0')

M_ext = build_marginalization_matrix_ext()

# Load all data from HDF5 to get ground truth text
h5_path = DATA_DIR / 'sentences_paper_256d.h5'
print(f"Loading data from {h5_path}...")
all_trials = []
with h5py.File(h5_path, 'r') as f:
    n_trials = f.attrs['n_trials']
    for i in range(n_trials):
        grp = f[f'trial_{i:05d}']
        t = {
            'features_raw': grp['features'][:],
            'phoneme_indices': grp['phoneme_indices'][:],
            'session': grp.attrs['session'],
            'text': grp.attrs.get('text', ''),
            'trial_idx': i,
        }
        all_trials.append(t)
        if (i+1) % 2000 == 0:
            print(f"  {i+1}/{n_trials}")

print(f"Loaded {len(all_trials)} trials, {sum(1 for t in all_trials if t['text'])} with text")

session_names = sorted(set(t['session'] for t in all_trials))
session_to_idx = {s: i for i, s in enumerate(session_names)}
for t in all_trials:
    t['session_idx'] = session_to_idx[t['session']]

sessions = {}
for t in all_trials:
    sessions.setdefault(t['session'], []).append(t)

test_sessions = session_names[-4:]
remaining = session_names[:-4]
val_sessions = remaining[-2:]
train_sessions = remaining[:-2]

train_sids = {session_to_idx[s] for s in train_sessions}
train_trials = [t for s in train_sessions for t in sessions[s]]
train_trials_with_text = [t for t in train_trials if t['text']]
print(f"Train trials with text: {len(train_trials_with_text)}")

# Decode with each of 4 seeds for data augmentation
seed_paths = [
    ('seed42', str(RESULTS_DIR / 'L2_L2_pe_dcond_adam_s42_long_best.pt')),
    ('seed43', str(RESULTS_DIR / 'L2_L2_pe_dcond_adam_s43_best.pt')),
    ('seed44', str(RESULTS_DIR / 'L2_L2_pe_dcond_adam_s44_best.pt')),
    ('seed45', str(RESULTS_DIR / 'L2_L2_pe_dcond_adam_s45_best.pt')),
]

train_data = []
batch_size = 32

for seed_name, ckpt_path in seed_paths:
    print(f"\n--- Decoding with {seed_name}: {ckpt_path} ---")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model = PaperExactDCoND(**ckpt['model_kwargs'])
    state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state)
    model.to(device).eval()
    model.set_train_sessions(train_sids)

    seed_data = []
    for i in range(0, len(train_trials_with_text), batch_size):
        batch = train_trials_with_text[i:i+batch_size]
        features = [torch.FloatTensor(t['features_raw']) for t in batch]
        feat_lens = [f.shape[0] for f in features]
        max_T = max(feat_lens)
        C = features[0].shape[1]
        padded = torch.zeros(len(batch), max_T, C).to(device)
        for j, f in enumerate(features):
            padded[j, :f.shape[0], :] = f
        session_ids = torch.LongTensor([t.get('session_idx', 0) for t in batch]).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(padded, session_ids)
            probs = logits.float().softmax(dim=-1).cpu()
            mono_probs = probs @ M_ext
            mono_lp = (mono_probs + 1e-10).log().numpy()

        for j, t in enumerate(batch):
            T_raw = feat_lens[j]
            T_out = max(1, (T_raw - model.kernel_size) // model.stride + 1)
            T_out = min(T_out, mono_lp.shape[1])
            decoded = ctc_greedy_decode(mono_lp[j, :T_out], blank=CTC_BLANK)
            target = t['phoneme_indices'].tolist()
            decoded_phones = ' '.join([CLASS_TO_ARPABET[p] for p in decoded])
            per = compute_per(decoded, target)

            seed_data.append({
                'decoded_phones': decoded_phones,
                'ground_truth': t['text'],
                'per': per,
                'seed': seed_name,
            })

    avg_per = np.mean([d['per'] for d in seed_data])
    print(f"  {seed_name}: {len(seed_data)} examples, PER={avg_per:.1%}")
    train_data.extend(seed_data)

    del model
    torch.cuda.empty_cache()

# Also add ground truth phoneme → text pairs (teaches the mapping without errors)
gt_data = []
for t in train_trials_with_text:
    target_phones = ' '.join([CLASS_TO_ARPABET[p] for p in t['phoneme_indices'].tolist()])
    gt_data.append({
        'decoded_phones': target_phones,
        'ground_truth': t['text'],
        'per': 0.0,
        'seed': 'ground_truth',
    })

print(f"\nGround truth pairs: {len(gt_data)}")
train_data.extend(gt_data)

# Shuffle
random.seed(42)
random.shuffle(train_data)
print(f"\nTotal training examples: {len(train_data)}")
print(f"  From 4 seeds: {len(train_data) - len(gt_data)}")
print(f"  From GT: {len(gt_data)}")

# Save training data
save_path = '/mnt/home/vincent.wilmet/brain2speech/data/phoneme_lift_v2_train.json'
with open(save_path, 'w') as f:
    json.dump(train_data, f)
print(f"Saved: {save_path}")

# Free GPU memory fully
torch.cuda.empty_cache()
import gc; gc.collect()

# ── Fine-tune Qwen ──
print("\n" + "="*60)
print("Fine-tuning Qwen on augmented phoneme→text data")
print("="*60)

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset

base_model_name = 'Qwen/Qwen3.5-2B'
tok = AutoTokenizer.from_pretrained(base_model_name)
tok.pad_token = tok.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, dtype=torch.bfloat16, device_map='cuda:0')

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64, lora_alpha=128, lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                     'gate_proj', 'up_proj', 'down_proj'],
)
peft_model = get_peft_model(base_model, lora_config)
trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in peft_model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

PROMPT = "Convert these decoded brain-computer interface phonemes to English text.\n\nPhonemes: {phones}\n\nEnglish:"

class PhonemeLiftDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.examples = []
        for ex in data:
            prompt = PROMPT.format(phones=ex['decoded_phones'])
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=False,
                add_generation_prompt=True)
            target_text = ex['ground_truth'].strip() + tokenizer.eos_token

            full_text = input_text + target_text
            enc = tokenizer(full_text, truncation=True, max_length=max_len,
                           padding='max_length', return_tensors='pt')

            input_ids = enc.input_ids[0]
            attention_mask = enc.attention_mask[0]

            input_enc = tokenizer(input_text, return_tensors='pt')
            input_len = input_enc.input_ids.shape[1]

            labels = input_ids.clone()
            labels[:input_len] = -100
            labels[attention_mask == 0] = -100

            self.examples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

print("Building dataset...")
dataset = PhonemeLiftDataset(train_data, tok)
print(f"Dataset: {len(dataset)} examples")

output_dir = '/mnt/home/vincent.wilmet/brain2speech/models/phoneme_lift_qwen_v2'
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=50,
    save_strategy='epoch',
    save_total_limit=2,
    dataloader_num_workers=0,
    report_to='none',
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
)

print("Starting training...")
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f"Training complete in {elapsed:.0f}s")

final_path = os.path.join(output_dir, 'final')
peft_model.save_pretrained(final_path)
tok.save_pretrained(final_path)
print(f"Model saved: {final_path}")
