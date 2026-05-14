import numpy as np; np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Simulate 5 drugs × 200 resistance mutations ──────────────────────────────
N_DRUGS = 5
N_MUTATIONS = 200
N_PATIENTS = 300

drug_names = ['Imatinib', 'Erlotinib', 'Vemurafenib', 'Olaparib', 'Pembrolizumab']
drug_targets = ['BCR-ABL', 'EGFR', 'BRAF', 'PARP1', 'PD-1']

# Resistance mutations per drug
mutations_per_drug = 40  # 5 × 40 = 200
mutation_names = []
for drug in drug_names:
    for i in range(mutations_per_drug):
        mutation_names.append(f'{drug[:3].upper()}_M{i+1:03d}')

drug_assignment = np.repeat(np.arange(N_DRUGS), mutations_per_drug)

# Fitness cost of resistance (growth rate relative to WT = 1.0)
# Most resistance mutations have some fitness cost
fitness_cost = np.random.beta(5, 2, N_MUTATIONS)  # 0-1, higher = less cost
fitness_cost = fitness_cost * 0.4 + 0.6  # scale to 0.6-1.0

# Some mutations are nearly cost-free (clinically important)
cost_free_idx = np.random.choice(N_MUTATIONS, 20, replace=False)
fitness_cost[cost_free_idx] = np.random.uniform(0.92, 1.0, 20)

# Drug-target binding affinity change (ΔΔG in kcal/mol)
# Positive ΔΔG = resistance (weaker binding)
ddg = np.random.lognormal(0.5, 0.8, N_MUTATIONS)  # mostly positive (resistance)
ddg = ddg * np.sign(np.random.normal(1, 0.3, N_MUTATIONS))  # some sensitizing
ddg = np.clip(ddg, -3, 8)

# Cross-resistance patterns (5×5 correlation matrix)
# Drugs with same target class have cross-resistance
cross_resistance_base = np.eye(N_DRUGS)
# Imatinib-Erlotinib: low cross-resistance (different targets)
# Vemurafenib-Erlotinib: moderate (both kinase inhibitors)
cross_resistance_base[1, 2] = cross_resistance_base[2, 1] = 0.45
cross_resistance_base[0, 1] = cross_resistance_base[1, 0] = 0.15
cross_resistance_base[0, 2] = cross_resistance_base[2, 0] = 0.20
cross_resistance_base[3, 4] = cross_resistance_base[4, 3] = 0.10
cross_resistance_base[1, 3] = cross_resistance_base[3, 1] = 0.25
cross_resistance_base += np.random.normal(0, 0.05, (N_DRUGS, N_DRUGS))
np.fill_diagonal(cross_resistance_base, 1.0)
cross_resistance_base = np.clip(cross_resistance_base, 0, 1)

# Resistance mutation co-occurrence (subset: 20 mutations)
n_cooccur = 20
cooccur_matrix = np.zeros((n_cooccur, n_cooccur))
for i in range(n_cooccur):
    for j in range(n_cooccur):
        if i == j:
            cooccur_matrix[i, j] = 1.0
        elif drug_assignment[i] == drug_assignment[j]:
            cooccur_matrix[i, j] = np.random.beta(2, 3)
        else:
            cooccur_matrix[i, j] = np.random.beta(0.5, 5)

# Evolutionary trajectory modeling
# Mutation order: which mutations appear first
n_trajectories = 5
trajectory_length = 6
trajectories = []
for t in range(n_trajectories):
    drug_idx = t % N_DRUGS
    drug_muts = np.where(drug_assignment == drug_idx)[0]
    # Order by fitness cost (least costly first)
    ordered = drug_muts[np.argsort(fitness_cost[drug_muts])[::-1]][:trajectory_length]
    trajectories.append(ordered)

# Clinical resistance threshold prediction
# Resistance frequency in patient population
resistance_freq = np.random.beta(2, 5, N_MUTATIONS)
# High-frequency mutations are clinically important
clinical_threshold = 0.10  # 10% frequency
clinical_resistant = resistance_freq > clinical_threshold

# Resistance mutation landscape (heatmap: drugs × top mutations)
n_top_per_drug = 8
landscape_matrix = np.zeros((N_DRUGS, n_top_per_drug))
for d in range(N_DRUGS):
    drug_muts = np.where(drug_assignment == d)[0]
    top_muts = drug_muts[np.argsort(resistance_freq[drug_muts])[::-1][:n_top_per_drug]]
    landscape_matrix[d] = resistance_freq[top_muts]

# ── Dashboard ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Drug Resistance Analysis Dashboard', fontsize=18,
             color='white', fontweight='bold', y=0.98)

DARK = '#161b22'
TEXT = 'white'
ACCENT = '#58a6ff'
ACCENT2 = '#f78166'
ACCENT3 = '#3fb950'

def style_ax(ax, title):
    ax.set_facecolor(DARK)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# 1. Resistance mutation landscape
ax = axes[0, 0]
style_ax(ax, '1. Resistance Mutation Landscape')
im = ax.imshow(landscape_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
ax.set_yticks(range(N_DRUGS))
ax.set_yticklabels([f'{drug_names[d]}\n({drug_targets[d]})' for d in range(N_DRUGS)],
                   color=TEXT, fontsize=7)
ax.set_xticks(range(n_top_per_drug))
ax.set_xticklabels([f'M{i+1}' for i in range(n_top_per_drug)], color=TEXT, fontsize=8)
ax.set_xlabel('Top Resistance Mutations (by frequency)', color=TEXT, fontsize=9)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(colors=TEXT, labelsize=7)
cbar.set_label('Resistance Frequency', color=TEXT, fontsize=8)

# 2. Fitness cost distribution
ax = axes[0, 1]
style_ax(ax, '2. Fitness Cost of Resistance Mutations')
colors_drug = ['#58a6ff', '#f78166', '#3fb950', '#d2a8ff', '#ffa657']
for d in range(N_DRUGS):
    drug_muts = np.where(drug_assignment == d)[0]
    ax.hist(fitness_cost[drug_muts], bins=15, alpha=0.6, color=colors_drug[d],
            label=drug_names[d], density=True)
ax.axvline(1.0, color='white', lw=1.5, ls='--', label='WT fitness')
ax.set_xlabel('Relative Fitness (growth rate)', color=TEXT, fontsize=9)
ax.set_ylabel('Density', color=TEXT, fontsize=9)
ax.legend(fontsize=7, facecolor='#21262d', labelcolor=TEXT)

# 3. ΔΔG heatmap (drugs × top mutations)
ax = axes[0, 2]
style_ax(ax, '3. Binding Affinity Change (ΔΔG) Heatmap')
ddg_matrix = np.zeros((N_DRUGS, n_top_per_drug))
for d in range(N_DRUGS):
    drug_muts = np.where(drug_assignment == d)[0]
    top_muts = drug_muts[np.argsort(ddg[drug_muts])[::-1][:n_top_per_drug]]
    ddg_matrix[d] = ddg[top_muts]
im = ax.imshow(ddg_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-2, vmax=6)
ax.set_yticks(range(N_DRUGS))
ax.set_yticklabels(drug_names, color=TEXT, fontsize=8)
ax.set_xticks(range(n_top_per_drug))
ax.set_xticklabels([f'M{i+1}' for i in range(n_top_per_drug)], color=TEXT, fontsize=8)
for i in range(N_DRUGS):
    for j in range(n_top_per_drug):
        ax.text(j, i, f'{ddg_matrix[i,j]:.1f}', ha='center', va='center',
                color='white', fontsize=6)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(colors=TEXT, labelsize=7)
cbar.set_label('ΔΔG (kcal/mol)', color=TEXT, fontsize=8)

# 4. Cross-resistance network
ax = axes[1, 0]
style_ax(ax, '4. Cross-Resistance Network')
angles = np.linspace(0, 2 * np.pi, N_DRUGS, endpoint=False)
x_pos = np.cos(angles)
y_pos = np.sin(angles)
for i in range(N_DRUGS):
    for j in range(i+1, N_DRUGS):
        cr = cross_resistance_base[i, j]
        if cr > 0.15:
            lw = cr * 5
            color = ACCENT2 if cr > 0.4 else ACCENT
            ax.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]],
                    color=color, alpha=0.7, lw=lw)
            mid_x = (x_pos[i] + x_pos[j]) / 2
            mid_y = (y_pos[i] + y_pos[j]) / 2
            ax.text(mid_x, mid_y, f'{cr:.2f}', ha='center', va='center',
                    color='yellow', fontsize=7)
ax.scatter(x_pos, y_pos, s=300, c=colors_drug, zorder=5, edgecolors='white', lw=2)
for i, (x, y) in enumerate(zip(x_pos, y_pos)):
    ax.text(x * 1.35, y * 1.35, drug_names[i], ha='center', va='center',
            color=TEXT, fontsize=8, fontweight='bold')
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.axis('off')
ax.set_facecolor(DARK)

# 5. Co-occurrence matrix
ax = axes[1, 1]
style_ax(ax, '5. Resistance Mutation Co-occurrence Matrix')
im = ax.imshow(cooccur_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(0, n_cooccur, 5))
ax.set_yticks(range(0, n_cooccur, 5))
ax.set_xticklabels([f'M{i}' for i in range(0, n_cooccur, 5)], color=TEXT, fontsize=7)
ax.set_yticklabels([f'M{i}' for i in range(0, n_cooccur, 5)], color=TEXT, fontsize=7)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(colors=TEXT, labelsize=7)
cbar.set_label('Co-occurrence Frequency', color=TEXT, fontsize=8)

# 6. Evolutionary trajectory
ax = axes[1, 2]
style_ax(ax, '6. Evolutionary Resistance Trajectories')
for t_idx, traj in enumerate(trajectories):
    drug_idx = t_idx % N_DRUGS
    fitness_traj = [1.0] + [fitness_cost[m] for m in traj]
    ddg_traj = [0.0] + [ddg[m] for m in traj]
    x = range(len(fitness_traj))
    ax.plot(x, fitness_traj, 'o-', color=colors_drug[drug_idx], lw=2, ms=6,
            label=drug_names[drug_idx], alpha=0.8)
ax.axhline(1.0, color='white', lw=1, ls='--', alpha=0.5)
ax.set_xlabel('Mutation Order (evolutionary step)', color=TEXT, fontsize=9)
ax.set_ylabel('Relative Fitness', color=TEXT, fontsize=9)
ax.legend(fontsize=7, facecolor='#21262d', labelcolor=TEXT)
ax.set_xticks(range(trajectory_length + 1))
ax.set_xticklabels(['WT'] + [f'Step {i+1}' for i in range(trajectory_length)], fontsize=7)

# 7. Clinical resistance threshold
ax = axes[2, 0]
style_ax(ax, '7. Clinical Resistance Threshold Prediction')
for d in range(N_DRUGS):
    drug_muts = np.where(drug_assignment == d)[0]
    ax.scatter(ddg[drug_muts], resistance_freq[drug_muts],
               c=colors_drug[d], alpha=0.6, s=20, label=drug_names[d])
ax.axhline(clinical_threshold, color='yellow', lw=2, ls='--',
           label=f'Clinical threshold ({clinical_threshold*100:.0f}%)')
ax.axvline(0, color='white', lw=1, ls=':', alpha=0.5)
ax.set_xlabel('ΔΔG (kcal/mol)', color=TEXT, fontsize=9)
ax.set_ylabel('Resistance Frequency', color=TEXT, fontsize=9)
ax.legend(fontsize=7, facecolor='#21262d', labelcolor=TEXT)
n_clinical = clinical_resistant.sum()
ax.text(0.97, 0.95, f'Clinical mutations: {n_clinical}', transform=ax.transAxes,
        ha='right', va='top', color=TEXT, fontsize=8,
        bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))

# 8. Resistance frequency by drug
ax = axes[2, 1]
style_ax(ax, '8. Resistance Mutation Frequency by Drug')
drug_freq_means = []
drug_freq_stds = []
for d in range(N_DRUGS):
    drug_muts = np.where(drug_assignment == d)[0]
    drug_freq_means.append(resistance_freq[drug_muts].mean())
    drug_freq_stds.append(resistance_freq[drug_muts].std())
bars = ax.bar(range(N_DRUGS), [f * 100 for f in drug_freq_means],
              yerr=[s * 100 for s in drug_freq_stds],
              color=colors_drug, edgecolor='#0d1117', alpha=0.85,
              error_kw=dict(ecolor='white', capsize=5))
ax.set_xticks(range(N_DRUGS))
ax.set_xticklabels(drug_names, rotation=30, fontsize=8, color=TEXT)
ax.set_ylabel('Mean Resistance Frequency (%)', color=TEXT, fontsize=9)
ax.axhline(clinical_threshold * 100, color='yellow', lw=1.5, ls='--', label='Clinical threshold')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor=TEXT)

# 9. Summary
ax = axes[2, 2]
style_ax(ax, '9. Analysis Summary')
ax.axis('off')
summary_lines = [
    ('Drugs Analyzed', f'{N_DRUGS}'),
    ('Resistance Mutations', f'{N_MUTATIONS}'),
    ('Clinical Mutations (>10%)', f'{n_clinical} ({n_clinical/N_MUTATIONS*100:.1f}%)'),
    ('Mean Fitness Cost', f'{(1-fitness_cost).mean():.3f}'),
    ('Cost-Free Mutations', f'{len(cost_free_idx)}'),
    ('Mean ΔΔG', f'{ddg.mean():.2f} kcal/mol'),
    ('Max ΔΔG', f'{ddg.max():.2f} kcal/mol'),
    ('Highest Cross-Resist.', f'{drug_names[1]}-{drug_names[2]}: {cross_resistance_base[1,2]:.2f}'),
    ('Most Resistant Drug', f'{drug_names[np.argmax(drug_freq_means)]}'),
    ('Evolutionary Steps', f'{trajectory_length}'),
]
y_pos = 0.95
for label, value in summary_lines:
    ax.text(0.05, y_pos, label + ':', color='#8b949e', fontsize=9, transform=ax.transAxes)
    ax.text(0.65, y_pos, value, color=ACCENT3, fontsize=9, fontweight='bold', transform=ax.transAxes)
    y_pos -= 0.09

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/mnt/shared-workspace/shared/drug_resistance_engine_dashboard.png',
            dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()

import shutil
try:
    shutil.copy(__file__, '/mnt/shared-workspace/shared/drug_resistance_engine.py')
except shutil.SameFileError:
    pass  # already in destination

print("=== DrugResistanceEngine Results ===")
print(f"Drugs: {N_DRUGS}, Mutations: {N_MUTATIONS}")
print(f"Clinical mutations (>10%): {n_clinical} ({n_clinical/N_MUTATIONS*100:.1f}%)")
print(f"Mean fitness cost: {(1-fitness_cost).mean():.3f}")
print(f"Mean ΔΔG: {ddg.mean():.2f} kcal/mol")
print(f"Cross-resistance matrix:\n{cross_resistance_base.round(2)}")
print(f"Drug resistance frequencies: {dict(zip(drug_names, [round(f*100,1) for f in drug_freq_means]))}")
print(f"Dashboard saved: /mnt/shared-workspace/shared/drug_resistance_engine_dashboard.png")
