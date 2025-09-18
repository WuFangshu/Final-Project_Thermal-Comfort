# Final Project: Thermal Comfort with Synthetic Data

This repository contains the implementation of Fangshu Wu’s final project on synthetic data generation for thermal comfort prediction. 
The project explores how synthetic data can support deep learning models when real-world human subject data is scarce, costly, or constrained by ethical issues.

## 📂 Repository Structure

Final-Project_Thermal-Comfort/
├── README.md
│ 
├──Download and convert
|
├── Long Short Term Memory training pipeline (LSTM)
│ 
├── Extended Long Short Term Memory training pipeline (xLSTM)
│ 
├── Generate Synthetic data witn Gaussium_Noise
│ 
├── Generate Synthetic data witn GAN
│ 
├── Generate Synthetic data witn LLM
│ 
├── Generate Synthetic data witn SNG
|
└── Compare synthetic data with raw data

## ⚙️ Environment Setup

Recommended local environment setup:

git clone https://github.com/WuFangshu/Final-Project_Thermal-Comfort.git

cd Final-Project_Thermal-Comfort

conda create -n thermal-comfort python=3.9 -y

conda activate thermal-comfort

*Each folder corresponds to the respective execution method.

pip install -r requirements.txt

