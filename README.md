Final Project: Thermal Comfort with Synthetic Data

This repository contains the implementation of Fangshu Wu’s final project on synthetic data generation for thermal comfort prediction. 
The project explores how synthetic data can support deep learning models when real-world human subject data is scarce, costly, or constrained by ethical issues.

⚙️ Setup Instructions
1. Clone the repository
git clone https://github.com/WuFangshu/Final-Project_Thermal-Comfort.git

cd Final-Project_Thermal-Comfort

3. Create a conda environment

conda create -n thermal-comfort python=3.9 -y

conda activate thermal-comfort

4. Install dependencies

Navigate to the LSTM folder and install the required packages:

cd LSTM

pip install -r requirements.txt

4. Download dataset

https://huggingface.co/datasets/kopetri/AutoTherm

Convert to CSV format.

Create a data/ directory inside each model folder (e.g., LSTM/data/, xLSTM/data/, etc.) and place the CSV files there.

🚀 Running the Models
LSTM

cd LSTM

Update dataset_path in train.py to the correct location

Rename your dataset splits to: aaa_training, aaa_test, aaa_validation

python train.py

xLSTM

cd ../xLSTM

Update dataset_path in train.py to the correct location

Ensure split names match: aaa_training, aaa_test, aaa_validation

python train.py

Gaussian Noise Augmentation

cd ../Gaussium_Noise

Update dataset_path in noise.py

python noise.py

Large Language Model (LLM)

cd ../LLM

Update dataset_path in llm.py

Add your API key in llm.py under API_KEY

python llm.py

Note: You need to rent a Qwen3 model from AIHubMix
.

GAN

cd ../GAN

Update dataset_path in gan.py

python gan.py


