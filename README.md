# 🩺 Kidney Disease Classification  

A **Deep Learning project** that classifies **Kidney CT Scan Images** into **Normal** or **CKD (Chronic Kidney Disease)** using **Convolutional Neural Networks (CNNs)**.  
This project aims to support early disease detection using AI-driven medical image analysis.  

---

## 🚀 Features  
- 🧠 **CNN-based Image Classifier** built with TensorFlow/Keras  
- 📊 Trains on kidney CT images to predict CKD vs Normal  
- 📈 Visualizations of training performance  
- 🖼️ Generates predictions with probability scores  

---

## 🛠️ Tech Stack  
**Languages & Libraries**  
- Python, NumPy, Pandas, Matplotlib, Seaborn  
- TensorFlow, Keras, Scikit-learn, OpenCV  

**ML/DL Techniques**  
- Convolutional Neural Networks (CNNs)  
- Data Augmentation (ImageDataGenerator)  
- Transfer Learning (optional, if used)  

---

## 📂 Project Structure  
```
Kidney_disease_classification/
│
├── .github/                # CI (workflows), PR and issue templates
│   └── workflows/
├── .dvc/                   # DVC metadata (already present)
├── .dvcignore
├── dvc.yaml                # DVC pipeline (already present)
├── dvc.lock                # DVC lockfile (already present)
├── params.yaml             # pipeline params (already present)
├── .gitignore
├── Dockerfile              # container image (already present)
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── pyproject.toml          # optional (packaging, lint/format config)
├── Makefile                # handy shortcuts: make data, make train, make test
│
├── data/                   # small control files, DVC tracks actual data
│   ├── README.md
│   ├── raw/                # immutable raw dataset (DVC-tracked)
│   ├── interim/            # intermediate, partially processed
│   └── processed/          # final datasets ready for modeling
│
├── models/                 # trained model artefacts (DVC-tracked)
│   ├── checkpoints/
│   └── production/         # final serialized model(s)
│
├── notebooks/              # exploratory analysis and EDA notebooks
│   ├── 01-eda.ipynb
│   └── 02-model-experiments.ipynb
│
├── research/               # experiments, paper notes (already present)
│   └── ...
│
├── src/                    # source code (package)
│   ├── __init__.py
│   ├── config/             # configuration loading utilities
│   │   └── config.py
│   ├── data/               # data loading & preprocessing
│   │   ├── make_dataset.py
│   │   └── preprocess.py
│   ├── features/           # feature engineering
│   │   └── build_features.py
│   ├── models/             # model definitions & training loops
│   │   ├── cnn_classifier.py        # move/merge from src/cnnClassifier
│   │   └── train.py                 # orchestrates training
│   ├── evaluation/         # evaluation metrics and plots
│   │   └── evaluate.py
│   ├── utils/              # logging, helpers, visualization
│   │   └── utils.py
│   └── cli.py              # command-line entrypoints (optional)
│
├── src/cnnClassifier/      # (current code) keep as package while refactoring
│   └── ...                 # short-term: move relevant files into src/models
│
├── app/                    # small web app / API (already have app.py)
│   ├── app.py              # flask/fastapi/streamlit entrypoint (you have app.py)
│   └── templates/          # HTML templates (already present)
│
├── scripts/                # helper scripts (dataset download, runs)
│   ├── download_data.sh
│   └── run_training.sh
│
├── tests/                  # unit + integration tests
│   ├── test_data.py
│   └── test_models.py
│
├── results/                # local generated plots, reports (ignore in git)
│   └── figures/
│
└── docs/                   # documentation (usage, design decisions)
    └── architecture.md

```

---

## 📊 Dataset  
- The dataset consists of **Kidney CT Scan Images** categorized into:  
  - **Normal**  
  - **CKD (Chronic Kidney Disease)**  

👉 *[Chronic Kidney Disease dataset](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)*  

---
# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/JunaidK0012/Kidney_disease_classification.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n venv python=3.8 -y
```

```bash
conda activate venv
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```

---


## 🔮 Future Work  
- Integrate into a **cloud-based medical AI pipeline** 
