# 🗂️ NIH ChestX-ray14 Dataset

This project uses the **NIH ChestX-ray14** dataset, one of the largest publicly available chest X-ray datasets for medical imaging research.

---

![Dataset_content](Dataset_content.png)

## 📌 Dataset Overview

- **Dataset Name**: NIH ChestX-ray14  
- **Total Images**: 112,120 chest X-rays  
- **Patients**: 30,805  
- **Image Type**: Frontal Chest X-rays (PA / AP)  
- **Size**: ~45 GB  
- **Annotations**: Image-level disease labels  
- **Source**: National Institutes of Health (NIH)

---

## 🏥 Disease Labels (14)

The dataset includes the following thoracic disease labels:

- Atelectasis  
- Cardiomegaly  
- Effusion  
- Infiltration  
- Mass  
- Nodule  
- Pneumonia  
- Pneumothorax  
- Consolidation  
- Edema  
- Emphysema  
- Fibrosis  
- Pleural Thickening  
- Hernia  
- No Finding

> ⚠️ Labels are **image-level**, not pixel-level.

---

## 📦 Dataset Access

Due to large size, the dataset is **NOT hosted in this repository**.

🔗 **Official Kaggle Link**:  
https://www.kaggle.com/datasets/nih-chest-xrays/data

---

## 📂 Original Dataset Structure

NIH-ChestXray/
├── images_001/
│ └── images/
├── images_002/
│ └── images/
├── ...
├── images_012/
│ └── images/
├── Data_Entry_2017.csv
├── train_val_list.txt
└── test_list.txt


---

## 🧠 How the Dataset is Used in ChestXpert-AI

- Images are paired with **synthetic radiology-style reports**
- Reports are generated based on disease labels
- Used for:
  - Vision-Language Model training
  - Report generation
  - Clinical reasoning evaluation

---

## 🔬 Data Splits

| Split | Usage |
|-----|------|
| Train | Model fine-tuning |
| Validation | Model selection |
| Test | Final evaluation only |

> Test data is **never used during training**.

---

## Preparing Dataset
![Preparing_Dataset](Preparing_Dataset.png)

## ⚠️ Dataset Limitations

- Labels may be noisy
- No bounding boxes or segmentation masks
- Single-view images only
- Reports are **synthetically generated** for training

---

## 📜 License & Usage

- Dataset provided for **research and educational purposes**
- Follow NIH & Kaggle usage guidelines
- Commercial usage may require additional permissions

---

## 🏥 Medical Disclaimer

This dataset is **NOT a diagnostic tool**.  
Always consult qualified medical professionals.

---

## 📚 Citation

If you use this dataset, please cite:

> Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM.  
> *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.*  
> CVPR 2017.
