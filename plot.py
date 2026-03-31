import matplotlib.pyplot as plt
import numpy as np
import json

from pathlib import Path

polarity_averages = []
weighted_sentiment_averages = []
files = [f.name for f in Path('.').iterdir() if f.is_file() and f.name.startswith('results')]


for file in files:
    with open(file, 'r') as f:
        content = json.load(f)
    polarity_values = [entry['sentiment']['polarity_index'] for entry in content]
    avg_polarity_value = np.mean(polarity_values)

    weighted_values = [entry['sentiment']['weighted_sentiment_score'] for entry in content]
    avg_weighted_value = np.mean(weighted_values)
    polarity_averages.append(avg_polarity_value)
    weighted_sentiment_averages.append(avg_weighted_value)

years = ['2022', '2023', '2024', '2025', '2026']
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
colors = ['#1F77B4', '#FF1E00', '#0082FA', '#FF5733', '#33FF57']

plots = [
    (polarity_averages, 'Average Polarity Index', 'Average Polarity Index: 2022-2026'),
    (weighted_sentiment_averages, 'Average Weighted Sentiment Score', 'Average Weighted Sentiment Score: 2022-2026')
]

for idx, (data, ylabel, title) in enumerate(plots):
    bars = axes[idx].bar(years, data, color=colors, alpha=0.8, width=0.5)
    axes[idx].set_ylabel(ylabel, fontsize=12, fontweight='bold')
    axes[idx].set_title(title, fontsize=14, fontweight='bold')
    axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[idx].grid(axis='y', alpha=0.3)
    axes[idx].set_ylim(-0.5, 0.5)
    
    for bar in bars:
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('sentiment_analysis_trends.png', dpi=300)
plt.show()


