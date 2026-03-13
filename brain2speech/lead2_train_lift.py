#!/usr/bin/env python3
"""Train Qwen to convert decoded phonemes directly to English text.

Instead of: phonemes → CMUDict words → LIFT → corrected text
This does:  phonemes → LIFT → English text directly

This avoids the lossy CMUDict greedy matching step.
"""
import json, sys, os, time
import numpy as np
import torch

sys.path.insert(0, 'brain2speech')

from config import CLASS_TO_ARPABET, N_CLASSES
from train_beyond_paper import (
    load_h5_dataset, compute_per, ctc_greedy_decode,
    DATA_DIR, RESULTS_DIR, CTC_BLANK,
)
from lead2_train_dcond import (
    PaperExactDCoND, build_marginalization_matrix_ext, N_MONO,
)

device = torch.device('cuda:0')

# ── Load best ensemble (seed 45) ──
ckpt_path = str(RESULTS_DIR / 'L2_L2_pe_dcond_adam_s45_best.pt')
print(f"Loading DCoND model: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
model = PaperExactDCoND(**ckpt['model_kwargs'])
state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict(state)
model.to(device).eval()

M_ext = build_marginalization_matrix_ext()

# ── Load data ──
h5_path = DATA_DIR / 'sentences_paper_256d.h5'
print(f"Loading data: {h5_path}")
trials = load_h5_dataset(h5_path, load_raw=True)

session_names = sorted(set(t['session'] for t in trials))
session_to_idx = {s: i for i, s in enumerate(session_names)}
for t in trials:
    t['session_idx'] = session_to_idx[t['session']]

sessions = {}
for t in trials:
    sessions.setdefault(t['session'], []).append(t)

test_sessions = session_names[-4:]
remaining = session_names[:-4]
val_sessions = remaining[-2:]
train_sessions = remaining[:-2]

train_sids = {session_to_idx[s] for s in train_sessions}
model.set_train_sessions(train_sids)

train_trials = [t for s in train_sessions for t in sessions[s]]
print(f"Train trials: {len(train_trials)}")

# ── Load ground truth text mapping ──
# Ground truth text from HDF5
import h5py
trial_texts = {}
with h5py.File(h5_path, 'r') as f:
    for i in range(f.attrs['n_trials']):
        grp = f[f'trial_{i:05d}']
        if 'sentence_text' in grp.attrs:
            trial_texts[i] = grp.attrs['sentence_text']

print(f"Trials with text: {len(trial_texts)}")

# ── Decode all training trials ──
print("\nDecoding training trials...")
train_data = []
batch_size = 32

for i in range(0, len(train_trials), batch_size):
    batch = train_trials[i:i+batch_size]
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
        target_phones = ' '.join([CLASS_TO_ARPABET[p] for p in target])
        
        # Get ground truth text
        gt_text = trial_texts.get(t.get('trial_idx', -1), '')
        if not gt_text:
            # Use target phonemes as fallback label  
            continue
        
        train_data.append({
            'decoded_phones': decoded_phones,
            'target_phones': target_phones,
            'ground_truth': gt_text,
        })
    
    if (i // batch_size) % 50 == 0:
        print(f"  {i+len(batch)}/{len(train_trials)} decoded ({len(train_data)} with text)")

print(f"\nTraining examples with text: {len(train_data)}")

# If we don't have sentence_text, use the training data we already generated
if len(train_data) < 100:
    print("No sentence_text in HDF5, loading existing training data...")
    with open('/mnt/home/vincent.wilmet/brain2speech/data/dcond_lift_train.json') as f:
        old_data = json.load(f)
    
    # Convert format: use decoded_phones directly, ground_truth as target
    train_data = []
    for ex in old_data:
        train_data.append({
            'decoded_phones': ex['decoded_phones'],
            'ground_truth': ex['ground_truth'],
        })
    print(f"  Loaded {len(train_data)} examples from existing data")

# Show examples
print("\nExamples:")
for ex in train_data[:3]:
    print(f"  Phones: {ex['decoded_phones'][:80]}")
    print(f"  Target: {ex['ground_truth'][:80]}")
    print()

# Save new training data
phoneme_lift_path = '/mnt/home/vincent.wilmet/brain2speech/data/phoneme_lift_train.json'
with open(phoneme_lift_path, 'w') as f:
    json.dump(train_data, f)
print(f"Saved {len(train_data)} examples to {phoneme_lift_path}")

# ── Free DCoND model memory ──
del model
torch.cuda.empty_cache()

# ── Fine-tune Qwen on phoneme→text ──
print("\n" + "="*60)
print("Fine-tuning Qwen on phoneme→text conversion")
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
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                     'gate_proj', 'up_proj', 'down_proj'],
)
peft_model = get_peft_model(base_model, lora_config)
trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in peft_model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

PHONEME_LIFT_PROMPT = "Convert these decoded brain-computer interface phonemes to English text.\n\nPhonemes: {phones}\n\nEnglish:"

class PhonemeLiftDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.examples = []
        for ex in data:
            prompt = PHONEME_LIFT_PROMPT.format(phones=ex['decoded_phones'])
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
            
            # Only compute loss on the target portion
            input_enc = tokenizer(input_text, return_tensors='pt')
            input_len = input_enc.input_ids.shape[1]
            
            labels = input_ids.clone()
            labels[:input_len] = -100  # Mask prompt
            labels[attention_mask == 0] = -100  # Mask padding
            
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

output_dir = '/mnt/home/vincent.wilmet/brain2speech/models/phoneme_lift_qwen'
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
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

# Save
final_path = os.path.join(output_dir, 'final')
peft_model.save_pretrained(final_path)
tok.save_pretrained(final_path)
print(f"Model saved: {final_path}")
