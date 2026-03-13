#!/usr/bin/env python3
"""Generate additional plots for Lead 1 writeup v2."""
import json
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PLOT_DIR = "brain2speech/results/lead1/plots"
import os
os.makedirs(PLOT_DIR, exist_ok=True)


# ── 7. Per-session analysis for best model ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Load best model history to extract per-epoch data
with open(glob.glob('brain2speech/results/lead1/L1.5h_cibr_cosine_20k_*.json')[0]) as f:
    d = json.load(f)
hist = d['history']

# Extract convergence speed: epochs to reach various PER thresholds
thresholds = [50, 40, 30, 25, 22, 21, 20, 19.5]
epochs_to_threshold = {}
for thresh in thresholds:
    for h in hist:
        if h['val_per'] * 100 <= thresh:
            epochs_to_threshold[thresh] = h['epoch']
            break

ax = axes[0]
if epochs_to_threshold:
    thresh_vals = sorted(epochs_to_threshold.keys(), reverse=True)
    epoch_vals = [epochs_to_threshold[t] for t in thresh_vals]
    ax.barh(range(len(thresh_vals)), epoch_vals, color='#377eb8', alpha=0.8)
    ax.set_yticks(range(len(thresh_vals)))
    ax.set_yticklabels([f'≤{t}%' for t in thresh_vals])
    ax.set_xlabel('Epochs Required')
    ax.set_title('L1.5h: Epochs to Reach PER Threshold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(epoch_vals):
        ax.text(v + 2, i, str(v), va='center', fontsize=9)

# Loss landscape: train loss vs val PER scatter for all experiments
ax = axes[1]
experiments_data = []
for f in sorted(glob.glob('brain2speech/results/lead1/*_*.json')):
    with open(f) as fh:
        dd = json.load(fh)
    name = f.split('/')[-1].split('_2026')[0]
    if dd.get('history') and len(dd['history']) > 10:
        final_loss = dd['history'][-1]['train_loss']
        val_per = dd['best_val_per'] * 100
        test_per = dd['test_per'] * 100
        hp = dd.get('hyperparams', {})
        sched = hp.get('scheduler', '?')
        opt = hp.get('optimizer', '?')
        if val_per < 35:  # filter out catastrophic runs
            color = '#ff7f00' if sched == 'cosine' else ('#4daf4a' if opt == 'sgd' else '#377eb8')
            marker = 'o' if sched == 'cosine' else ('s' if opt == 'sgd' else '^')
            ax.scatter(final_loss, test_per, c=color, marker=marker, s=60, alpha=0.7, zorder=3)
            if val_per < 20 or test_per > 24:
                ax.annotate(name.replace('L1.',''), (final_loss, test_per),
                           fontsize=6, alpha=0.7, xytext=(3, 3),
                           textcoords='offset points')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f00', markersize=8, label='SGD + Cosine'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#4daf4a', markersize=8, label='SGD + Step'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#377eb8', markersize=8, label='Adam'),
]
ax.legend(handles=legend_elements, fontsize=9)
ax.set_xlabel('Final Training Loss')
ax.set_ylabel('Test PER (%)')
ax.set_title('Train Loss vs Test PER (All Experiments)')
ax.grid(True, alpha=0.3)
ax.axhline(y=19.7, color='red', linestyle='--', alpha=0.4, linewidth=1)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/07_convergence_and_landscape.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 07_convergence_and_landscape.png")


# ── 8. Improvement waterfall chart ──
fig, ax = plt.subplots(figsize=(12, 6))

steps = [
    ('Baseline\n(k=1, wrong opt)', 51.9, 0),
    ('Fix arch\n(k=14, UniGRU)', 25.4, 0),
    ('BiGRU\n(k=32, h=1024)', 21.4, 0),
    ('SGD\n(lr=0.1, ortho)', 20.9, 0),
    ('Cosine 20K\nscheduler', 20.2, 0),
    ('+ Beam search\n(trigram LM)', 19.5, 0),
    ('+ 5-model\nensemble', 17.8, 0),
]

labels = [s[0] for s in steps]
values = [s[1] for s in steps]
x = np.arange(len(labels))

# Color based on improvement
colors = []
for i, v in enumerate(values):
    if i == 0:
        colors.append('#e41a1c')
    elif v <= 19.7:
        colors.append('#4daf4a')
    else:
        colors.append('#377eb8')

bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
ax.axhline(y=19.7, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Paper target (19.7%)')

# Add improvement arrows between bars
for i in range(1, len(values)):
    improvement = values[i-1] - values[i]
    if improvement > 0:
        ax.annotate(f'−{improvement:.1f}pp',
                   xy=(i, values[i] + 0.3),
                   fontsize=8, ha='center', color='#333333', fontweight='bold')

for i, v in enumerate(values):
    ax.text(i, v + 1.2 if i > 0 else v + 1.2, f'{v:.1f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Test PER (%)', fontsize=12)
ax.set_title('Lead 1 Improvement Waterfall: 51.9% → 17.8% Test PER', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 55)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/08_improvement_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 08_improvement_waterfall.png")


# ── 9. Cosine vs step LR decay detail ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Load step decay and cosine 20K
configs = {
    'Step Decay (L1.3)': ('L1.3_cibr_withinday', '#e41a1c'),
    'Cosine 20K (L1.5h)': ('L1.5h_cibr_cosine_20k', '#377eb8'),
}

for label, (prefix, color) in configs.items():
    files = glob.glob(f'brain2speech/results/lead1/{prefix}_*.json')
    if not files:
        continue
    with open(files[0]) as f:
        dd = json.load(f)
    hist = dd['history']
    epochs = [h['epoch'] for h in hist]
    val_per = [h['val_per'] * 100 for h in hist]
    train_loss = [h['train_loss'] for h in hist]
    lr = [h['lr'] for h in hist]

    # Top left: LR
    axes[0, 0].plot(epochs, lr, color=color, label=label, linewidth=2)
    # Top right: Train loss
    axes[0, 1].plot(epochs, train_loss, color=color, label=label, linewidth=1.5)
    # Bottom left: Val PER
    axes[1, 0].plot(epochs, val_per, color=color, label=label, linewidth=1.5)
    # Bottom right: Val PER (zoomed)
    axes[1, 1].plot(epochs, val_per, color=color, label=label, linewidth=1.5)

axes[0, 0].set_title('Learning Rate')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('LR')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_title('Training Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('CTC Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(0, 2)

axes[1, 0].set_title('Validation PER')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Val PER (%)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(18, 40)

axes[1, 1].set_title('Validation PER (Zoomed)')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Val PER (%)')
axes[1, 1].axhline(y=19.7, color='gray', linestyle='--', alpha=0.5, label='Paper (19.7%)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(18.5, 22)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/09_step_vs_cosine_detail.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 09_step_vs_cosine_detail.png")


# ── 10. Seed diversity training curves ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

seed_colors = {'42': '#e41a1c', '123': '#377eb8', '456': '#4daf4a', '789': '#ff7f00'}
for seed, color in seed_colors.items():
    files = glob.glob(f'brain2speech/results/lead1/L1.7c_cos20k_seed{seed}_*.json')
    if not files:
        continue
    with open(files[0]) as f:
        dd = json.load(f)
    hist = dd['history']
    epochs = [h['epoch'] for h in hist]
    val_per = [h['val_per'] * 100 for h in hist]
    train_loss = [h['train_loss'] for h in hist]

    axes[0].plot(epochs, train_loss, color=color, label=f'Seed {seed}', alpha=0.7, linewidth=1.2)
    axes[1].plot(epochs, val_per, color=color, label=f'Seed {seed}', alpha=0.7, linewidth=1.2)

# Also plot L1.5h for comparison
files = glob.glob('brain2speech/results/lead1/L1.5h_cibr_cosine_20k_*.json')
if files:
    with open(files[0]) as f:
        dd = json.load(f)
    hist = dd['history']
    epochs = [h['epoch'] for h in hist]
    val_per = [h['val_per'] * 100 for h in hist]
    train_loss = [h['train_loss'] for h in hist]
    axes[0].plot(epochs, train_loss, color='black', label='L1.5h (original)', alpha=0.5,
                linewidth=2, linestyle='--')
    axes[1].plot(epochs, val_per, color='black', label='L1.5h (original)', alpha=0.5,
                linewidth=2, linestyle='--')

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('CTC Loss')
axes[0].set_title('Training Loss — Seed Diversity (Cosine 20K)')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation PER (%)')
axes[1].set_title('Validation PER — Seed Diversity (Cosine 20K)')
axes[1].axhline(y=19.7, color='gray', linestyle='--', alpha=0.5, label='Paper (19.7%)')
axes[1].legend(fontsize=9)
axes[1].set_ylim(18, 30)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/10_seed_diversity_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 10_seed_diversity_curves.png")


# ── 11. Architecture diversity: h=768 vs h=1024 vs h=1280 (cosine 20K) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

arch_configs = {
    'h=768 (57.6M)': ('L1.8a_h768_cos20k', '#4daf4a'),
    'h=1024 (98.3M)': ('L1.5h_cibr_cosine_20k', '#377eb8'),
    'h=1280 (149.9M)': ('L1.8c_h1280_cos20k', '#e41a1c'),
}

for label, (prefix, color) in arch_configs.items():
    files = glob.glob(f'brain2speech/results/lead1/{prefix}_*.json')
    if not files:
        continue
    with open(files[0]) as f:
        dd = json.load(f)
    hist = dd['history']
    epochs = [h['epoch'] for h in hist]
    val_per = [h['val_per'] * 100 for h in hist]
    train_loss = [h['train_loss'] for h in hist]

    axes[0].plot(epochs, train_loss, color=color, label=label, linewidth=1.5)
    axes[1].plot(epochs, val_per, color=color, label=label, linewidth=1.5)

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('CTC Loss')
axes[0].set_title('Training Loss — Hidden Size Comparison (Cosine 20K)')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation PER (%)')
axes[1].set_title('Validation PER — Hidden Size Comparison (Cosine 20K)')
axes[1].axhline(y=19.7, color='gray', linestyle='--', alpha=0.5, label='Paper (19.7%)')
axes[1].legend(fontsize=10)
axes[1].set_ylim(18, 30)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/11_hidden_size_cosine.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 11_hidden_size_cosine.png")


# ── 12. Full beam search sweep heatmap (extended) ──
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Val PER heatmap for beta=0.0
beam_widths = [5, 10]
alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
data_b0 = np.array([
    [19.21, 19.00, 18.82, 18.74, 18.81, 19.01, 19.51, 20.89, 22.69],
    [19.14, 18.94, 18.81, 18.73, 18.77, 19.00, 19.43, 20.58, 22.28],
])

# Val PER heatmap for beta=0.5
data_b5 = np.array([
    [19.21, 19.20, 18.89, 18.76, 18.75, 18.79, 19.20, 20.38, 22.06],
    [19.14, 19.16, 18.84, 18.77, 18.68, 18.73, 19.13, 20.17, 21.72],
])

for ax, data, beta_label in [(axes[0], data_b0, 'β=0.0'), (axes[1], data_b5, 'β=0.5')]:
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=18.5, vmax=23)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'{a:.1f}' for a in alphas])
    ax.set_yticks(range(len(beam_widths)))
    ax.set_yticklabels([f'beam={b}' for b in beam_widths])
    ax.set_xlabel('LM Weight (α)')
    ax.set_title(f'Val PER (%) — {beta_label}')

    for i in range(len(beam_widths)):
        for j in range(len(alphas)):
            color = 'white' if data[i, j] > 20.5 else 'black'
            fontw = 'bold' if data[i, j] == data.min() else 'normal'
            ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                   fontsize=9, color=color, fontweight=fontw)

plt.colorbar(im, ax=axes, label='PER (%)', shrink=0.8)
plt.suptitle('Beam Search Hyperparameter Sweep (L1.7c seed 789)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/12_beam_sweep_extended.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 12_beam_sweep_extended.png")


# ── 13. Ensemble composition analysis ──
fig, ax = plt.subplots(figsize=(10, 7))

ensemble_data = [
    ('Best single\n(greedy)', 20.2, '#e41a1c'),
    ('Best single\n(beam)', 19.5, '#e41a1c'),
    ('4-seed ensemble\n(beam)', 19.2, '#377eb8'),
    ('3-seed+L1.5h\n(beam)', 18.1, '#4daf4a'),
    ('3-seed+L1.5h\n+h=768 (beam)', 17.8, '#ff7f00'),
    ('6-model diverse\n(beam)', 18.4, '#984ea3'),
    ('5-all+s456\n(beam)', 18.7, '#a65628'),
    ('3-seed+L1.5h\n+h=1280 (beam)', 18.0, '#f781bf'),
]

names = [e[0] for e in ensemble_data]
values = [e[1] for e in ensemble_data]
colors = [e[2] for e in ensemble_data]

y_pos = np.arange(len(names))
bars = ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor='white', height=0.7)
ax.axvline(x=19.7, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Paper (19.7%)')
ax.axvline(x=17.8, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Best (17.8%)')

for i, v in enumerate(values):
    ax.text(v + 0.1, i, f'{v:.1f}%', va='center', fontsize=10, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel('Test PER (%)', fontsize=12)
ax.set_title('Ensemble Composition Study — Test PER', fontsize=14)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(16.5, 21)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/13_ensemble_composition.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 13_ensemble_composition.png")


# ── 14. Parameter efficiency plot ──
fig, ax = plt.subplots(figsize=(10, 6))

# Data: (params_M, test_per, label, color)
models = [
    (29, 25.4, 'Paper UniGRU\n(h=512, k=14)', '#e41a1c'),
    (57.6, 21.3, 'h=768 step', '#984ea3'),
    (57.6, 19.7, 'h=768 cosine', '#ff7f00'),
    (98.3, 21.4, 'cffan Adam', '#377eb8'),
    (98.3, 20.9, 'CIBR SGD step', '#4daf4a'),
    (98.3, 20.2, 'CIBR SGD cosine', '#ff7f00'),
    (149.9, 21.0, 'h=1280 step', '#984ea3'),
    (149.9, 20.3, 'h=1280 cosine', '#ff7f00'),
]

for params, per, label, color in models:
    ax.scatter(params, per, c=color, s=120, alpha=0.8, zorder=3, edgecolors='white', linewidth=1)
    ax.annotate(label, (params, per), fontsize=7, alpha=0.8,
               xytext=(5, 5), textcoords='offset points')

ax.axhline(y=19.7, color='red', linestyle='--', alpha=0.5, label='Paper target (19.7%)')
ax.set_xlabel('Parameters (M)', fontsize=12)
ax.set_ylabel('Test PER (%)', fontsize=12)
ax.set_title('Parameter Efficiency: Model Size vs Performance', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(18.5, 26)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/14_parameter_efficiency.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 14_parameter_efficiency.png")

print("\nAll v2 plots generated successfully!")
