import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

df = pd.read_csv("bank-full.csv", sep=";")
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['y'].value_counts()}\n")


X = df.drop("y", axis=1)
y = (df["y"] == "yes").astype(int)

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]


preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", "passthrough", num_cols),
])


model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model),
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print("=" * 50)
print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
print("=" * 50)
print(classification_report(y_test, pred, target_names=["No", "Yes"]))


ohe_features = (
    pipeline.named_steps["preprocess"]
    .named_transformers_["cat"]
    .get_feature_names_out(cat_cols)
    .tolist()
)
all_feature_names = ohe_features + num_cols

fig, ax = plt.subplots(figsize=(28, 10))
plot_tree(
    pipeline.named_steps["model"],
    feature_names=all_feature_names,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8,
    ax=ax,
)
ax.set_title("Decision Tree Visualization (Depth=3) — Bank Marketing Dataset", fontsize=14)
plt.tight_layout()
plt.savefig("Decision_Tree_Visualization.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tree visualization saved → Decision_Tree_Visualization.png")


cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
fig2, ax2 = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax2, colorbar=False, cmap="Blues")
ax2.set_title("Confusion Matrix — Bank Marketing Decision Tree")
plt.tight_layout()
plt.savefig("Confusion_Matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Confusion matrix saved  → Confusion_Matrix.png")


rules = export_text(
    pipeline.named_steps["model"],
    feature_names=all_feature_names,
    max_depth=3,
)
print("\nTop Decision Rules:\n")
print(rules)
