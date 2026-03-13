#!/usr/bin/env python3
"""Train DCoND-LIFT corrector: given (LIFT output + phonemes) → correct text.
This follows the paper's approach of feeding BOTH text AND phonemes to the LLM."""
import json, sys, time, re, os, random
import numpy as np
import h5py
import torch
from Levenshtein import distance as lev_distance

sys.path.insert(0, 'brain2speech')
from config import CLASS_TO_ARPABET, N_CLASSES
from train_beyond_paper import compute_per, ctc_greedy_decode, DATA_DIR, RESULTS_DIR, CTC_BLANK
from lead2_train_dcond import PaperExactDCoND, build_marginalization_matrix_ext

device = torch.device('cuda:0')

# Load data
h5_path = DATA_DIR / 'sentences_paper_256d.h5'
print("Loading data...")
trials = []
with h5py.File(h5_path, 'r') as f:
    n_trials = f.attrs['n_trials']
    for i in range(n_trials):
        grp = f[f'trial_{i:05d}']
        trials.append({
            'features_raw': grp['features'][:],
            'phoneme_indices': grp['phoneme_indices'][:],
            'session': grp.attrs['session'],
            'text': grp.attrs.get('text', ''),
        })

session_names = sorted(set(t['session'] for t in trials))
session_to_idx = {s: i for i, s in enumerate(session_names)}
for t in trials:
    t['session_idx'] = session_to_idx[t['session']]
sessions = {}
for t in trials:
    sessions.setdefault(t['session'], []).append(t)

train_sessions = session_names[:-6]
train_sids = {session_to_idx[s] for s in train_sessions}
train_trials = [t for s in train_sessions for t in sessions[s]]
M_ext = build_marginalization_matrix_ext()

# Step 1: Decode train set with seed45 to get phoneme sequences
print("Decoding train set with seed45...")
ckpt_path = str(RESULTS_DIR / 'L2_L2_pe_dcond_adam_s45_best.pt')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
model = PaperExactDCoND(**ckpt['model_kwargs'])
state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict(state)
model.to(device).eval()
model.set_train_sessions(train_sids)

train_phones = []
for i in range(0, len(train_trials), 64):
    batch = train_trials[i:i+64]
    features = [torch.FloatTensor(t['features_raw']) for t in batch]
    feat_lens = [f.shape[0] for f in features]
    max_T = max(feat_lens)
    C = features[0].shape[1]
    padded = torch.zeros(len(batch), max_T, C).to(device)
    for j, f in enumerate(features):
        padded[j, :f.shape[0], :] = f
    session_ids = torch.LongTensor([t.get('session_idx', 0) for t in batch]).to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        logits = model(padded, session_ids)
    probs = logits.float().softmax(dim=-1).cpu()
    mono_probs = probs @ M_ext
    mono_lp = (mono_probs + 1e-10).log().numpy()
    for j, t in enumerate(batch):
        T_raw = feat_lens[j]
        T_out = max(1, (T_raw - model.kernel_size) // model.stride + 1)
        T_out = min(T_out, mono_lp.shape[1])
        decoded = ctc_greedy_decode(mono_lp[j, :T_out], blank=CTC_BLANK)
        train_phones.append(' '.join([CLASS_TO_ARPABET[p] for p in decoded]))

del model; torch.cuda.empty_cache()
print(f"  Decoded {len(train_phones)} train trials")

# Step 2: Run LIFT v1 on train set to get initial translations
print("Loading LIFT v1...")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_path = '/mnt/home/vincent.wilmet/brain2speech/models/phoneme_lift_qwen/final'
tok = AutoTokenizer.from_pretrained(adapter_path)
lift_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-2B', torch_dtype=torch.bfloat16, device_map='cuda:0')
lift_model = PeftModel.from_pretrained(lift_model, adapter_path)
lift_model.eval()

PROMPT = "Convert these decoded brain-computer interface phonemes to English text.\n\nPhonemes: {phones}\n\nEnglish:"

print("Getting LIFT v1 translations for train set...")
lift_translations = []
t0 = time.time()
for idx, phones in enumerate(train_phones):
    gt = train_trials[idx]['text']
    if not gt:
        lift_translations.append('')
        continue
    prompt = PROMPT.format(phones=phones)
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt').to(lift_model.device)
    with torch.no_grad():
        out = lift_model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
    response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    lift_translations.append(response)
    if (idx + 1) % 500 == 0:
        elapsed = time.time() - t0
        print(f"  {idx+1}/{len(train_phones)} | {elapsed:.0f}s", flush=True)

del lift_model; torch.cuda.empty_cache()

# Step 3: Build DCoND-LIFT training data
# Format: (LIFT_output + phonemes) → ground_truth
CORRECTOR_PROMPT = """Correct this brain-computer interface transcription using both the decoded text and raw phonemes.

Decoded text: {text}
Phonemes: {phones}

Corrected text:"""

print("\nBuilding corrector training data...")
corrector_data = []
for idx in range(len(train_phones)):
    gt = train_trials[idx]['text']
    if not gt or not lift_translations[idx]:
        continue
    
    prompt = CORRECTOR_PROMPT.format(text=lift_translations[idx], phones=train_phones[idx])
    corrector_data.append({
        'prompt': prompt,
        'response': gt,
        'lift_text': lift_translations[idx],
        'phones': train_phones[idx],
    })

print(f"  {len(corrector_data)} corrector training examples")

# Step 4: Fine-tune Qwen with LoRA for correction
print("\nFine-tuning corrector model...")
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader

base_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-2B', torch_dtype=torch.bfloat16, device_map='cuda:0')

lora_config = LoraConfig(
    r=32, lora_alpha=64, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                          'gate_proj', 'up_proj', 'down_proj'],
    lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
corrector_model = get_peft_model(base_model, lora_config)
corrector_model.print_trainable_parameters()

class CorrectorDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        messages = [
            {"role": "user", "content": item['prompt']},
            {"role": "assistant", "content": item['response']}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
        
        # Find where response starts
        user_messages = [{"role": "user", "content": item['prompt']}]
        user_text = self.tokenizer.apply_chat_template(user_messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)
        user_len = len(self.tokenizer(user_text, return_tensors='pt').input_ids[0])
        
        encoded = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, 
                                  truncation=True, padding='max_length')
        input_ids = encoded.input_ids[0]
        attention_mask = encoded.attention_mask[0]
        
        labels = input_ids.clone()
        labels[:user_len] = -100  # Don't compute loss on prompt
        labels[attention_mask == 0] = -100  # Don't compute loss on padding
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

dataset = CorrectorDataset(corrector_data, tok)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.AdamW(corrector_model.parameters(), lr=1e-4, weight_decay=0.01)
corrector_model.train()

n_epochs = 3
for epoch in range(n_epochs):
    total_loss = 0
    n_batches = 0
    t0 = time.time()
    for batch in loader:
        optimizer.zero_grad()
        outputs = corrector_model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device))
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(corrector_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    elapsed = time.time() - t0
    print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {total_loss/n_batches:.4f} | {elapsed:.0f}s")

# Save
save_dir = '/mnt/home/vincent.wilmet/brain2speech/models/dcond_lift_corrector'
os.makedirs(save_dir, exist_ok=True)
corrector_model.save_pretrained(os.path.join(save_dir, 'final'))
tok.save_pretrained(os.path.join(save_dir, 'final'))
print(f"\nSaved corrector to {save_dir}/final")
