from collections import defaultdict

import pandas as pd
from tqdm import tqdm

gt_dir = '../../PaddleOCR_Mukhtar/train_data/rec/synthtiger'


def convert(root):
    # read the gt.txt file
    with open(f'{root}/gt.txt', 'r') as file:
        lines = file.readlines()

    # process the lines and group by directory
    data = defaultdict(list)
    for line in tqdm(lines):
        parts = line.strip().split('\t')
        path_parts = parts[0].split('/')
        directory = '/'.join(path_parts[:-1])
        filename = path_parts[-1]
        words = parts[1]
        data[directory].append([filename, words])

    # create a DataFrame and write to CSV for each directory
    for directory, rows in data.items():
        df = pd.DataFrame(rows, columns=['filename', 'words'])
        csv_filename = f'{root}/{directory}/labels.csv'
        df.to_csv(csv_filename, index=False)


def main():
    convert(f'{gt_dir}/train_more')
    convert(f'{gt_dir}/test')


if __name__ == '__main__':
    main()
