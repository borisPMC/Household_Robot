import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the data
# Replace 'file.csv' with your file path and 'label1', 'label2' with your column names
df = pd.read_csv('newer_data.csv')

# Step 2: Create a contingency table
heatmap_data = pd.crosstab(df['Language'], df['Label'])

# Step 3: Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Distribution of Class vs Language in Custom Dataset")
plt.xlabel("Label")
plt.ylabel("Language")
plt.savefig(fname="meta.png")
