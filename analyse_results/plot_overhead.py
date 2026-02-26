import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

df_q = pd.read_csv('quantized_profile.csv')
df_q['Precision'] = 'Quantized'
df_bf16 = pd.read_csv('full_model_bf16.csv')
df_bf16['Precision'] = 'BF16'
df = pd.concat([df_q, df_bf16], ignore_index=True)
df = df[df['Status'] == 'Success']

df['Calc_Time_Overhead(%)'] = ((df['Time_Strat(s)'] - df['Time_Base(s)']) / df['Time_Base(s)']) * 100
df['Calc_Res_Overhead(%)'] = ((df['Res_Strat(MB)'] - df['Res_Base(MB)']) / df['Res_Base(MB)']) * 100

df['Batch_str'] = pd.Categorical(df['Batch'].astype(str), categories=['4', '16', '64'], ordered=True)
df['Length_str'] = pd.Categorical(df['Length'].astype(str), categories=['8', '32', '128'], ordered=True)
df['Steps_str'] = pd.Categorical(df['Steps'].astype(str), categories=['4', '8', '16'], ordered=True)

sns.set_theme(style="white")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.5], hspace=0.3, wspace=0.35)

ax_top = fig.add_subplot(gs[0, :])
df_melt = df.melt(id_vars=['Precision'],
                  value_vars=['Calc_Time_Overhead(%)', 'Calc_Res_Overhead(%)'],
                  var_name='Metric', value_name='Overhead (%)')
df_melt['Metric'] = df_melt['Metric'].map({
    'Calc_Time_Overhead(%)': 'Time Overhead (%)',
    'Calc_Res_Overhead(%)': 'VRAM Overhead (%)'
})

sns.barplot(data=df_melt, x='Precision', y='Overhead (%)', hue='Metric',
            ax=ax_top, palette=['tab:blue', 'tab:green'], capsize=.05, errorbar=('ci', 95))

ax_top.set_title('Average Overhead', fontsize=15, pad=15)
ax_top.set_ylabel('Overhead (%)', fontsize=13, labelpad=10)
ax_top.set_xlabel('')
ax_top.tick_params(axis='both', labelsize=12)
ax_top.grid(True, axis='y', alpha=0.3)
ax_top.legend(title='', fontsize=12, loc='upper left')

x_vars = ['Batch_str', 'Length_str', 'Steps_str']
x_labels = ['Batch Size ($B$)', 'Sequence Length ($S$)', 'Diffusion Steps ($T$)']

colors = {'BF16': 'tab:blue', 'Quantized': 'tab:orange'}
axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(3)]

for i in range(3):
    ax1 = axes_bottom[i]
    ax2 = ax1.twinx()

    sns.lineplot(data=df, x=x_vars[i], y='Calc_Time_Overhead(%)', hue='Precision',
                 palette=colors, marker='o', linestyle='-', ax=ax1, errorbar=('ci', 95), legend=False)

    sns.lineplot(data=df, x=x_vars[i], y='Calc_Res_Overhead(%)', hue='Precision',
                 palette=colors, marker='s', err_style='bars', ax=ax2, errorbar=('ci', 95), legend=False)

    for line in ax2.lines:
        line.set_linestyle('--')

    ax1.set_xlabel(x_labels[i], fontsize=13, labelpad=10)

    if i == 0:
        ax1.set_ylabel('Time Overhead (%)', fontsize=13, labelpad=10)
    else:
        ax1.set_ylabel('')

    if i == 2:
        ax2.set_ylabel('VRAM Overhead (%)', fontsize=13, labelpad=10)
    else:
        ax2.set_ylabel('')

    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)

    ax1.grid(True, axis='both', alpha=0.3)

custom_lines = [
    Line2D([0], [0], color=colors['BF16'], marker='o', linestyle='-', label='Time (BF16)'),
    Line2D([0], [0], color=colors['Quantized'], marker='o', linestyle='-', label='Time (Quantized)'),
    Line2D([0], [0], color=colors['BF16'], marker='s', linestyle='--', label='VRAM (BF16)'),
    Line2D([0], [0], color=colors['Quantized'], marker='s', linestyle='--', label='VRAM (Quantized)')
]

fig.legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False, fontsize=13)

plt.savefig('overhead_combined_categorical.pdf', format='pdf', dpi=300, bbox_inches='tight')