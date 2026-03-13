#!/usr/bin/env python3
"""Resume LIFT v2 training with multi-GPU support."""
import json, sys, os, time, random
import numpy as np
import torch

sys.path.insert(0, 'brain2speech')
from train_beyond_paper import DATA_DIR, RESULTS_DIR

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.utils.data import Dataset

# Load training data (already generated)
train_data_path = '/mnt/home/vincent.wilmet/brain2speech/data/phoneme_lift_v2_train.json'
print(f"Loading training data from {train_data_path}...")
with open(train_data_path) as f:
    train_data = json.load(f)
print(f"Loaded {len(train_data)} examples")

base_model_name = 'Qwen/Qwen3.5-2B'
tok = AutoTokenizer.from_pretrained(base_model_name)
tok.pad_token = tok.eos_token

# For multi-GPU, don't use device_map
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, torch_dtype=torch.bfloat16)

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

# More GPUs = larger effective batch size, so adjust
n_gpus = torch.cuda.device_count()
print(f"Using {n_gpus} GPUs")

# Keep effective batch size the same: 4 * 4 = 16 per GPU
# With N GPUs: per_device=4, grad_accum = max(1, 16 // (4 * n_gpus))
grad_accum = max(1, 16 // (4 * n_gpus))

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=grad_accum,
    learning_rate=1e-4,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_strategy='epoch',
    save_total_limit=3,
    dataloader_num_workers=2,
    report_to='none',
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
)

# Resume from latest checkpoint if exists
ckpt_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
if ckpt_dirs:
    latest = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))[-1]
    resume_path = os.path.join(output_dir, latest)
    print(f"Resuming from {resume_path}")
    trainer.train(resume_from_checkpoint=resume_path)
else:
    print("Starting fresh training")
    trainer.train()

elapsed = time.time()
print(f"Training complete")

final_path = os.path.join(output_dir, 'final')
peft_model.save_pretrained(final_path)
tok.save_pretrained(final_path)
print(f"Model saved: {final_path}")
