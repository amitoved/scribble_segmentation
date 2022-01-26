kaggle datasets download -d anisayari/siimacrpneumothoraxsegmentationzip-dataset
unzip "archive.zip"
python scribble_segmentation/generate_demo_set.py --config config_siim_acr.txt