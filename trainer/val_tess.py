import pytesseract
from PIL import Image
import pandas as pd
import os
import yaml
from utils import AttrDict
from tqdm import tqdm

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    return AttrDict(opt)

def load_labels(csv_path):
    df = pd.read_csv(csv_path, sep='\t', engine='python', usecols=['filename', 'words'], keep_default_na=False)
    # take only 10
    # df = df.sample(n=100, random_state=1)
    return dict(zip(df.filename, df.words))

def evaluate_tesseract(data_path, labels):
    total_images = len(labels)
    correct_predictions = 0

    for filename, ground_truth in tqdm(labels.items(), disable=False):
        image_path = os.path.join(data_path, filename)
        image = Image.open(image_path)
        recognized_text = pytesseract.image_to_string(image, lang="kaz")

        # Simple comparison, can be improved with more sophisticated metrics
        if recognized_text.strip() == ground_truth.strip():
            correct_predictions += 1
        # else:
        #     print(f"Ground truth: {ground_truth}")
        #     print(f"Recognized text: {recognized_text}")
        

    accuracy = correct_predictions / total_images
    return accuracy

def main():
    opt = get_config("config_files/best_accuracy.yaml")
    pref_path = os.path.join(opt.valid_data, opt.select_data)
    labels = load_labels(os.path.join(pref_path, 'labels.csv'))
    accuracy = evaluate_tesseract(pref_path, labels)
    print(f"Tesseract OCR Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
