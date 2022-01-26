import requests
import os
from typing import List


def download_data(langs: List[str],
                  force: bool = False,
                  version: str = '4.0.0'):
    """Downloads training and testing data for a set of languages from Rik van
    Noort's GitHub repository.

    Args:
        langs (List[str]): [description] force (bool): [description]
    """
    print('Downloading...')
    base_url = 'https://raw.githubusercontent.com/RikVN/DRS_parsing/master/parsing/layer_data/{version}'
    for lang in langs:
        save_path = f'./data/{lang}'
        for fname in ['train.conll', 'test.conll', 'dev.conll', 'eval.conll']:
            url = f'{base_url}/{lang}/gold/{fname}'
            os.makedirs(save_path, exist_ok=True)

            if os.path.exists(f'{save_path}/{fname}') and not force:
                print(f'{fname : <11} for {lang} found, skipping download')
            else:
                resp = requests.get(url)
                if not resp.ok:
                    print(
                        f'File {fname} does not exist for {lang}. skipping...')
                else:
                    with open(f'{save_path}/{fname}', 'w') as file:
                        file.write(resp.text)

        # Create merged train/dev set
        with open(f'{save_path}/all.conll', 'w') as all_txt:
            with open(f'{save_path}/train.conll', 'r') as train_txt,\
                 open(f'{save_path}/dev.conll', 'r') as test_txt:
                all_txt.write(train_txt.read())
                all_txt.write(test_txt.read())
            # Try to attach eval.conll if it exists (doesn't for all languages)
            try:
                with open(f'{save_path}/eval.conll', 'r') as eval_txt:
                    all_txt.write(eval_txt.read())
            except IOError:
                continue  # Just ignore error if eval.conll doesn't exist

    print('Finished downloading.')
