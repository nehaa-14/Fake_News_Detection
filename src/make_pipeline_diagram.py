import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

stages = [
    "Raw Tweets\n(CONSTRAINT COVID-19\nfake news dataset)",
    "Text Cleaning\n(lowercase, strip URLs,\n@mentions, punctuation)",
    "Feature Engineering\n(TF-IDF, 1-2 grams,\nstopword removal)",
    "Model Training\n(Logistic Regression,\nLinear SVM, Naive Bayes)",
    "Evaluation\n(Accuracy, Precision,\nRecall, F1, ROC-AUC)",
    "Best Model Selected\n(highest F1 on\nheld-out test set)",
]

fig, ax = plt.subplots(figsize=(14, 3.2))
ax.set_xlim(0, len(stages))
ax.set_ylim(0, 1)
ax.axis("off")

colors = ["#EAF2FB", "#DCEAF7", "#CFE3F3", "#C2DBEE", "#B5D3EA", "#9FC7E4"]

for i, (stage, color) in enumerate(zip(stages, colors)):
    box = FancyBboxPatch((i + 0.06, 0.25), 0.88, 0.5,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          linewidth=1.3, edgecolor="#2C5F8A", facecolor=color)
    ax.add_patch(box)
    ax.text(i + 0.5, 0.5, stage, ha="center", va="center", fontsize=9.5, color="#1B3A56")
    if i < len(stages) - 1:
        arrow = FancyArrowPatch((i + 0.95, 0.5), (i + 1.05, 0.5),
                                 arrowstyle="-|>", mutation_scale=18, color="#2C5F8A", linewidth=1.5)
        ax.add_patch(arrow)

plt.tight_layout()
plt.savefig("/home/claude/fake_news_project/outputs/figures/pipeline_diagram.png", dpi=170, bbox_inches="tight")
print("saved")
