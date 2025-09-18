# Thermal Comfort Classification
Pytorch based models for classification of thermal comfort states using a multi-modal data.

# Excution:

conda activate Final-Project_Thermal-Comfort

Put raw .csv data in parquedata folder

Make this line "parser.add_argument('--dataset_path', default="E:/autotherm-3CFF/parquetdata", type=str, help="Path to dataset.")" to correct local path.

Make sure dataloader/splits/aaa_training.txt, dataloader/splits/aaa_test.txt, dataloader/splits/aaa_validation.txt correspond to parquedata folder

python lstm.py
