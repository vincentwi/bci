#!/usr/bin/env python3
"""Generate plots for Lead 1 writeup."""
import json
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = "brain2speech/results/lead1/plots"
import os
os.makedirs(PLOT_DIR, exist_ok=True)

# ── 1. Training curves for key experiments ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

experiments = {
    'L1.1c (Paper UniGRU)': 'L1.1c_paper_withinday',
    'L1.2b (cffan BiGRU)': 'L1.2b_cffan_withinday',
    'L1.3 (CIBR SGD)': 'L1.3_cibr_withinday',
    'L1.4 (Linderman)': 'L1.4_linderman_withinday',
}

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

for (label, prefix), color in zip(experiments.items(), colors):
    files = glob.glob(f'brain2speech/results/lead1/{prefix}_*.json')
    if not files:
        continue
    with open(files[0]) as f:
        d = json.load(f)
    hist = d['history']
    epochs = [h['epoch'] for h in hist]
    val_per = [h['val_per'] * 100 for h in hist]
    train_loss = [h['train_loss'] for h in hist]

    axes[0].plot(epochs, train_loss, color=color, label=label, alpha=0.8, linewidth=1.5)
    axes[1].plot(epochs, val_per, color=color, label=label, alpha=0.8, linewidth=1.5)

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('CTC Loss')
axes[0].set_title('Training Loss — Literature Reproductions')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 4)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation PER (%)')
axes[1].set_title('Validation PER — Literature Reproductions')
axes[1].axhline(y=19.7, color='gray', linestyle='--', alpha=0.5, label='Paper target (19.7%)')
axes[1].legend(fontsize=9)
axes[1].set_ylim(15, 50)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/01_literature_reproductions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 01_literature_reproductions.png")


# ── 2. Scheduler comparison ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sched_exps = {
    'Step Decay (L1.3)': ('L1.3_cibr_withinday', '#e41a1c'),
    'Cosine 100K (L1.5f)': ('L1.5f_cibr_cosine', '#377eb8'),
    'Cosine 200K (L1.5g)': ('L1.5g_cibr_cosine_long', '#4daf4a'),
    'Cosine 20K (L1.5h)': ('L1.5h_cibr_cosine_20k', '#ff7f00'),
}

for label, (prefix, color) in sched_exps.items():
    files = glob.glob(f'brain2speech/results/lead1/{prefix}_*.json')
    if not files:
        continue
    with open(files[0]) as f:
        d = json.load(f)
    hist = d['history']
    epochs = [h['epoch'] for h in hist]
    val_per = [h['val_per'] * 100 for h in hist]
    lr = [h['lr'] for h in hist]

    axes[0].plot(epochs, lr, color=color, label=label, alpha=0.8, linewidth=1.5)
    axes[1].plot(epochs, val_per, color=color, label=label, alpha=0.8, linewidth=1.5)

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Learning Rate')
axes[0].set_title('Learning Rate Schedule Comparison')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation PER (%)')
axes[1].set_title('Validation PER — Scheduler Comparison')
axes[1].axhline(y=19.7, color='gray', linestyle='--', alpha=0.5, label='Paper (19.7%)')
axes[1].legend(fontsize=9)
axes[1].set_ylim(17, 35)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/02_scheduler_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 02_scheduler_comparison.png")


# ── 3. Ablation bar chart ──
fig, ax = plt.subplots(figsize=(12, 6))

ablations = [
    ('Paper\nUniGRU h=512 k=14', 25.4, '#e41a1c'),
    ('cffan\nBiGRU h=1024 k=32', 21.4, '#377eb8'),
    ('CIBR\nSGD step', 20.9, '#4daf4a'),
    ('Linderman\nh=512 post-RNN', 23.8, '#984ea3'),
    ('Enhanced\nall techniques', 24.7, '#ff7f00'),
    ('CIBR k=14', 21.6, '#a65628'),
    ('CIBR h=768', 21.3, '#f781bf'),
    ('CIBR h=1280', 21.0, '#999999'),
    ('CIBR noise=1.0', 22.0, '#e41a1c'),
    ('CIBR 7 layers', 34.5, '#377eb8'),
    ('CIBR 3 layers', 21.3, '#4daf4a'),
    ('CIBR cosine\n100K', 20.4, '#984ea3'),
    ('CIBR cosine\n20K ★', 20.2, '#ff7f00'),
    ('CIBR +\nsupTCon', 21.0, '#a65628'),
]

names = [a[0] for a in ablations]
values = [a[1] for a in ablations]
bar_colors = [a[2] for a in ablations]

bars = ax.bar(range(len(names)), values, color=bar_colors, alpha=0.8, edgecolor='white')
ax.axhline(y=19.7, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Paper target (19.7%)')
ax.axhline(y=20.2, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Best single (20.2%)')

ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Test PER (%)')
ax.set_title('Lead 1 Ablation Study — Test PER')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(values):
    ax.text(i, v + 0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/03_ablation_study.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 03_ablation_study.png")


# ── 4. Seed diversity + ensemble ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Seed comparison
seeds_val = {'42': 19.5, '123': 19.5, '456': 20.2, '789': 19.3}
seeds_test = {'42': 20.2, '123': 20.2, '456': 21.1, '789': 20.3}

x = list(seeds_val.keys())
ax = axes[0]
width = 0.35
x_pos = np.arange(len(x))
ax.bar(x_pos - width/2, [seeds_val[s] for s in x], width, label='Val PER', color='#377eb8', alpha=0.8)
ax.bar(x_pos + width/2, [seeds_test[s] for s in x], width, label='Test PER', color='#e41a1c', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Seed {s}' for s in x])
ax.set_ylabel('PER (%)')
ax.set_title('Seed Diversity — Individual Model Performance')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(18, 22)

# Ensemble comparison
ensembles = [
    ('Best Single\n(greedy)', 19.3, 20.2),
    ('Best Single\n(beam)', 18.8, 19.5),
    ('4-seed\nensemble', 18.7, 19.2),
    ('4-model\n(L1.5h+seeds)', 17.7, 18.1),
    ('5-model\n(+h=768) ★', 17.5, 17.8),
    ('6-model\n(+h=1280)', 17.7, 18.0),
]

ax = axes[1]
names = [e[0] for e in ensembles]
val_pers = [e[1] for e in ensembles]
test_pers = [e[2] for e in ensembles]
x_pos = np.arange(len(names))

ax.bar(x_pos - width/2, val_pers, width, label='Val PER', color='#377eb8', alpha=0.8)
ax.bar(x_pos + width/2, test_pers, width, label='Test PER', color='#e41a1c', alpha=0.8)
ax.axhline(y=19.7, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='Paper (19.7%)')
ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel('PER (%)')
ax.set_title('Ensemble Ablation — Val & Test PER')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(16, 21)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/04_ensemble_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 04_ensemble_results.png")


# ── 5. Beam search heatmap ──
fig, ax = plt.subplots(figsize=(10, 6))

# Data from the beam search sweep on seed 789
beam_widths = [5, 10]
alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
# Val PER values (beta=0.0)
data = np.array([
    [19.21, 19.00, 18.82, 18.74, 18.81, 19.01, 19.51, 20.89, 22.69],  # beam=5
    [19.14, 18.94, 18.81, 18.73, 18.77, 19.00, 19.43, 20.58, 22.28],  # beam=10
])

im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=18.5, vmax=23)
ax.set_xticks(range(len(alphas)))
ax.set_xticklabels([f'{a:.1f}' for a in alphas])
ax.set_yticks(range(len(beam_widths)))
ax.set_yticklabels([f'beam={b}' for b in beam_widths])
ax.set_xlabel('LM Weight (α)')
ax.set_title('Beam Search Val PER (%) — β=0.0')

for i in range(len(beam_widths)):
    for j in range(len(alphas)):
        color = 'white' if data[i, j] > 20 else 'black'
        ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center', fontsize=9, color=color)

plt.colorbar(im, label='PER (%)')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/05_beam_search_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 05_beam_search_heatmap.png")


# ── 6. Best model training curve (L1.5h) ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

with open(glob.glob('brain2speech/results/lead1/L1.5h_cibr_cosine_20k_*.json')[0]) as f:
    d = json.load(f)
hist = d['history']
epochs = [h['epoch'] for h in hist]
val_per = [h['val_per'] * 100 for h in hist]
train_loss = [h['train_loss'] for h in hist]
lr = [h['lr'] for h in hist]

axes[0].plot(epochs, train_loss, color='#377eb8', linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('CTC Loss')
axes[0].set_title('L1.5h Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, val_per, color='#e41a1c', linewidth=1.5)
axes[1].axhline(y=19.3, color='green', linestyle='--', alpha=0.7, label='Best (19.3%)')
axes[1].axhline(y=19.7, color='gray', linestyle='--', alpha=0.5, label='Paper (19.7%)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation PER (%)')
axes[1].set_title('L1.5h Validation PER')
axes[1].legend()
axes[1].set_ylim(17, 40)
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs, lr, color='#4daf4a', linewidth=1.5)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].set_title('L1.5h Cosine LR Schedule')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/06_best_model_training.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 06_best_model_training.png")

print("\nAll plots generated successfully!")
