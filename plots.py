import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Seaborn style for better aesthetics
sns.set_theme(style="whitegrid")

# Path to your CSV files (adjust as needed)
input_folder = "results/"

# List all CSV files in the directory
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# Choose the metrics to plot
metrics = ["MRR@3", "Precision@3", "Hits@3", "Cosine Similarity"]

# Create output folder for plots
output_folder = "plots/"
os.makedirs(output_folder, exist_ok=True)

for csv_file in csv_files:
    # Load the CSV file
    file_path = os.path.join(input_folder, csv_file)
    df = pd.read_csv(file_path, sep=";")
    
    # Extract experiment name from file name
    experiment_name = os.path.splitext(csv_file)[0]
    
    # Plot each metric separately
    for metric in metrics:
        if metric in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x="Noise Level", y=metric, hue="Experiment Type")
            plt.title(f"{metric} by Noise Level and Experiment Type - {experiment_name}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_file = os.path.join(output_folder, f"{experiment_name}_{metric}_boxplot.png")
            plt.savefig(plot_file)
            plt.close()
            print(f"Saved box plot: {plot_file}")

            # Create bar plot for the same metric
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="Noise Level", y=metric, hue="Experiment Type", ci=None)
            plt.title(f"{metric} by Noise Level and Experiment Type - {experiment_name}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            bar_plot_file = os.path.join(output_folder, f"{experiment_name}_{metric}_barplot.png")
            plt.savefig(bar_plot_file)
            plt.close()
            print(f"Saved bar plot: {bar_plot_file}")

            # Scatter plot for metric relationships
            if "Cosine Similarity" in df.columns and metric != "Cosine Similarity":
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df, x="Cosine Similarity", y=metric, hue="Experiment Type")
                plt.title(f"{metric} vs Cosine Similarity - {experiment_name}")
                plt.tight_layout()
                scatter_file = os.path.join(output_folder, f"{experiment_name}_{metric}_scatter.png")
                plt.savefig(scatter_file)
                plt.close()
                print(f"Saved scatter plot: {scatter_file}")

print("All experiment plots have been generated and saved.")
