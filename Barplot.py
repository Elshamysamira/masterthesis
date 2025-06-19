import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data for Avg. Cosine Similarity — 0.6 Threshold
data = [
    ("OpenAI", "EN–EN", 0.8780),
    ("OpenAI", "EN*–EN", 0.864),
    ("OpenAI", "DE–EN", 0.8377),
    ("OpenAI", "DE*–EN", 0.8305),

    ("Cohere", "EN–EN", 0.6751),
    ("Cohere", "EN*–EN", 0.645),
    ("Cohere", "DE–EN", 0.5936),
    ("Cohere", "DE*–EN", 0.5774),

    ("LaBSE", "EN–EN", 0.5055),
    ("LaBSE", "EN*–EN", 0.469),
    ("LaBSE", "DE–EN", 0.4827),
    ("LaBSE", "DE*–EN", 0.4515),
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Model", "Setting", "Avg. Cosine Similarity"])

# Create bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Avg. Cosine Similarity", hue="Setting", data=df)

# Formatting
plt.title("Average Cosine Similarity across Models and Settings — Threshold 0.6")
plt.ylabel("Avg. Cosine Similarity")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.legend(title="Setting", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Display
plt.show()
