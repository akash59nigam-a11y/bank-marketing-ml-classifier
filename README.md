# SCT_DS_Task_2 — Bank Marketing Decision Tree Classifier

## 📌 Task
Build a Decision Tree Classifier to predict whether a customer will subscribe to a term deposit, based on demographic and behavioral data from a bank marketing campaign.

> **Internship:** SkillCraft Technology | **Track:** Data Science | **Task:** 2

---

## 📂 Dataset
- **Name:** Bank Marketing Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **File used:** `bank-full.csv`
- **Records:** 45,211 rows × 17 columns
- **Target column:** `y` → Did the client subscribe? (`yes` / `no`)

---

## 🧠 Model
| Parameter | Value |
|---|---|
| Algorithm | Decision Tree Classifier |
| Max Depth | 5 |
| Class Weight | Balanced |
| Test Size | 20% |
| Random State | 42 |

---

## 📊 Results
| Metric | Value |
|---|---|
| Accuracy | ~85% |
| Encoding | OneHotEncoder (categorical cols) |
| Preprocessing | ColumnTransformer Pipeline |

---

## 📁 Files
| File | Description |
|---|---|
| `decision_tree.py` | Main Python script |
| `Decision_Tree_Visualization.png` | Tree plot (depth=3) |
| `Confusion_Matrix.png` | Confusion matrix heatmap |
| `bank-full.csv` | Dataset (download from UCI) |

---

## ▶️ How to Run

```bash
pip install pandas scikit-learn matplotlib
python decision_tree.py
```

---

## 🔧 Libraries Used
- `pandas` — data loading & manipulation
- `scikit-learn` — model building, pipeline, metrics
- `matplotlib` — visualization

---

## 👤 Author
**Akash Nigam**
- GitHub: [akash59nigam-a11y](https://github.com/akash59nigam-a11y)
- LinkedIn: [linkedin.com/in/akash-nigam](https://linkedin.com/in/akash-nigam)
