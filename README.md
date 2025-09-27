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
├── config/
│   └── config.yaml
├── research/               # experiments
│   ├── 01_data_ingestion.ipynb
│   └── trials.ipynb
│
├── src/cnnClassifier                  # source code 
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   └── data_ingestion.py
│   │   └── model_evaluation_mlflow.py
│   │   └── model_training.py
│   │   └── prepare_base_model.py
│   ├── config/             # configuration loading utilities
│   │   ├── __init__.py
│   │   └── configuration.py
│   ├── constants/             
│   │   ├── __init__.py
│   ├── entity/
│   │   ├── __init__.py
│   │   └── config_entity.py
│   ├── pipeline/
│   │   ├── __init__.py         
│   │   ├── prediction.py        
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_model_training.py
│   │   ├── stage_04_model_evaluation.py
│   ├── utils/              # logging, helpers, visualization
│   │   ├── __init__.py 
│   │   └── common.py
├── app.py              # flask entrypoint 
├── templates/          # HTML templates
├── .dvcignore
├── dvc.yaml                # DVC pipeline (already present)
├── dvc.lock                # DVC lockfile (already present)
├── params.yaml             # pipeline params (already present)
├── .gitignore
├── Dockerfile              # container image (already present)
├── README.md
├── requirements.txt
├── setup.py 

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
