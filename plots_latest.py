import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ─── SETTINGS ────────────────────────────────────────────────────────────────
input_folder = "plots/openai/openai_06_GER_ENG_NQCD"
files = {
    "light":   "german_english_noisy_queries_clean_documents_light_results.csv",
    "moderate":"german_english_noisy_queries_clean_documents_moderate_results.csv",
    "severe":  "german_english_noisy_queries_clean_documents_severe_results.csv"
    #"clean_clir": "german_english_clean_queries_clean_documents_results.csv"
    #"monolingual_de": "german_german_clean_queries_clean_documents_results.csv"
}
output_folder = os.path.join(input_folder, "plots_comparison_by_noise")
os.makedirs(output_folder, exist_ok=True)

metrics = ["Cosine Similarity", "MRR@3", "Precision@3", "Hits@3"]

# ─── LOAD & COMBINE ──────────────────────────────────────────────────────────
dfs = []
for level, fname in files.items():
    path = os.path.join(input_folder, fname)
    df = pd.read_csv(path, sep=";")
    df["Noise Level"] = level
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

# ─── PLOTTING ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.stripplot(
        data=combined,
        x="Noise Level",
        y=metric,
        hue="Noise Level",
        dodge=True,
        jitter=0.25,
        alpha=0.7,
        size=6
    )
    plt.title(f"{metric} by Noise Level")
    plt.xlabel("Noise Level")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"scatter_{metric.replace('@','at')}.png"))
    plt.close()

print(f"Scatter plots saved under: {output_folder}")
