# Plate‐Component Classification Pipeline

A two‐stage workflow for extracting features from license‐plate data, exploratory analysis, and building a baseline classification model using CatBoost (and friends).

---

## 🚀 Project Workflow

1. **Data Exploration & Feature Engineering**  
   - Extract raw plate components  
   - Inspect distributions, correlations, missingness  
   - Build new features (e.g. length, character‐type ratios, region codes)  
   - See `data_exploration.ipynb`

2. **Automated Column Detection & Modeling**  
   - Load the engineered dataset  
   - Automatically identify numeric vs. categorical columns  
   - Train and evaluate baseline classifiers (LightGBM, XGBoost, CatBoost)  
   - Inspect logs and metrics in `catboost_info/`  
   - See `data_modeling.ipynb`

3. **Generate Submission**  
   - Apply your best‐performing model to `test.csv`  
   - Write predictions to `submission.csv`  

---

## 📂 Repository Structure

```

.
├── catboost\_info/                # CatBoost training artifacts & logs
│   ├── catboost\_training.json
│   ├── learn/
│   │   └── events.out.tfevents   # TensorBoard logs
│   ├── learn\_error.tsv           # per‐iteration error
│   ├── time\_left.tsv             # ETA per iteration
│   └── tmp/                      # intermediate CatBoost files
├── data/                         # raw & processed datasets + helpers
│   ├── processed\_train.pkl       # after feature engineering
│   ├── processed\_test.pkl
│   ├── supplemental\_english.py   # language‐specific helpers
│   ├── supplemental\_russian.py
│   ├── train.csv                 # raw training data
│   ├── test.csv                  # raw test data
├── data\_exploration.ipynb        # feature‐engineering & EDA
├── data\_modeling.ipynb           # automated column detection & modeling
├── submission.csv                # your final predictions
└── README.md                     # this file

````

---

## 🤖 Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/theadityamittal/russian-car-plates.git
   cd russian-car-plates
   ````

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   If there is no `requirements.txt`, install manually:

   ```bash
   pip install numpy pandas scikit-learn lightgbm xgboost catboost jupyter matplotlib
   ```

---

## ▶️ Usage

1. **Launch Jupyter**

   ```bash
   jupyter notebook
   ```

2. **Run Notebooks in Order**

   * **`data_exploration.ipynb`**

     * Cleans and merges raw CSVs
     * Generates `processed_train.pkl` & `processed_test.pkl` in `data/`
   * **`data_modeling.ipynb`**

     * Reads the processed pickles
     * Auto‐detects column types
     * Trains CatBoost (and optionally LightGBM/XGBoost)
     * Saves logs under `catboost_info/`
     * Outputs predictions to `submission.csv`

---

## 📈 Outputs & Artifacts

* **`data/processed_*.pkl`**: engineered features
* **`catboost_info/`**: training logs, metrics, and TensorBoard events
* **`submission.csv`**: your final predictions in “ID,label” format

---

## 📄 Data Description

* **`train.csv` / `test.csv`**: raw license-plate strings, target labels
* **`supplemental_{english,russian}.py`**: helper functions for language‐specific parsing

---

## 🤝 Contributing

* Feel free to open an issue to discuss improvements
* Submit pull requests for bug fixes, new feature ideas, or clearer documentation

---

## 📜 License

This project is released under the MIT License.

