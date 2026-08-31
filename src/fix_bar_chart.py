import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir('/Users/kadirkesgin/Documents/akademikcalismalar/2026/mart2026/education_truth')

labels = ['Baseline Model', 'Proposed Method']
low_ses = [47.75, 8.08]
high_ses = [43.53, 13.23]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(x - width/2, low_ses, width, label='Low SES Burden', color='#E63946')
rects2 = ax.bar(x + width/2, high_ses, width, label='High SES Burden', color='#2A9D8F')

ax.set_ylabel('Average Intervention Effort (Cost)', fontsize=12)
# ax.set_title removed for LaTeX
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=11)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('final_optimization_results.png', dpi=300)
print("Grafik başarıyla 'Gruplandırılmış Maliyet Grafiği' olarak güncellendi.")
