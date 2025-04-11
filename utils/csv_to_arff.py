import pandas as pd

csv_path = "datasets/synthetic_rand.csv"
arff_sparse_path = "datasets/synthetic_rand.arff"

# Load CSV
df = pd.read_csv(csv_path)

# Define feature vs label columns
label_cols = [col for col in df.columns if col.startswith("y")]
feature_cols = [col for col in df.columns if col not in label_cols]
all_cols = feature_cols + label_cols  # order matters for index mapping

with open(arff_sparse_path, "w") as f:
    f.write("@RELATION synthetic_monolab_sparse\n\n")

    # Write attributes
    for col in feature_cols:
        f.write(f"@ATTRIBUTE {col} NUMERIC\n")
    for col in label_cols:
        f.write(f"@ATTRIBUTE {col} {{0,1}}\n")

    f.write("\n@DATA\n")

    # Write data in sparse format: only non-zero values as {index value}
    for _, row in df.iterrows():
        entries = []
        for i, val in enumerate(row[all_cols]):
            if val != 0:
                entries.append(f"{i} {val}")
        f.write("{" + ", ".join(entries) + "}\n")
