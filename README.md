Final Project: Thermal Comfort with Synthetic Data

This repository contains the implementation of Fangshu Wu’s final project on synthetic data generation for thermal comfort prediction. 
The project explores how synthetic data can support deep learning models when real-world human subject data is scarce, costly, or constrained by ethical issues.

Setup

git clone https://github.com/WuFangshu/Final-Project_Thermal-Comfort.git,
cd Final-Project_Thermal-Comfort

Create virtual environment on Terminal:

conda create -n thermal-comfort python=3.9 -y,
conda activate thermal-comfort

cd LSTM,
pip install -r requirements.txt

Download Datasets on https://huggingface.co/datasets/kopetri/AutoTherm,
Convert to CSV files,
Create a data folder in each folder and place the CSV files inside.

cd LSTM,
Modify the dataset_path in train.py to the correct path.
Rename the files aaa_training, aaa_test, and aaa_validation in the dataloader/split folder to match the names in the created data folder.
python train.py

cd .. ,
cd xLSTM,
Modify the dataset_path in train.py to the correct path.
Rename the files aaa_training, aaa_test, and aaa_validation in the dataloader/split folder to match the names in the created data folder.
python train.py

cd .. ,
cd noise,
Modify the dataset_path in noise.py to the correct path.
python noise.py

cd .. ,
cd LLM ,
Modify the dataset_path in llm.py to the correct path.
Rent a model on https://aihubmix.com, enter the key into the API Key field in llm.py.
python llm.py

cd .. ,
cd GAN ,
Modify the dataset_path in gan.py to the correct path.
python gan.py ,

cd .. ,
cd SNG ,
Modify the dataset_path in sng_autotherm_from_csv.py to the correct path.
python sng_autotherm_from_csv.py

