import os
import sys
import requests
import argparse
from typing import List
from baseline.DecisionTreeTagger import DecisionTreeTagger, \
                                        DecisionTreeTaggerOptions
from baseline.SvmTagger import SvmClassifier
from baseline.TrigramTagger import TrigramTagger
from baseline.HiddenMarkovModel import HmmTagger
from tagger.RNNTagger import RnnTagger


def download_data(langs: List[str],
                  force: bool = False,
                  version: str = '4.0.0'):
    """Downloads training and testing data for a set of languages from Rik van
    Noort's GitHub repository.

    Args:
        langs (List[str]): [description] force (bool): [description]
    """
    print('Downloading...')
    base_url = f'https://raw.githubusercontent.com/RikVN/DRS_parsing/master/parsing/layer_data/{version}'
    for lang in langs:
        for fname in ['train.conll', 'test.conll']:
            url = f'{base_url}/{lang}/gold/{fname}'
            save_path = f'./data/{lang}'
            os.makedirs(save_path, exist_ok=True)

            if os.path.exists(f'{save_path}/{fname}') and not force:
                print(f'{fname : <11} for {lang} found, skipping download')
            else:
                resp = requests.get(url)
                if not resp.ok:
                    print(
                        'Download failed. Is the repository still available?')
                    print(resp)
                    exit()
                else:
                    with open(f'{save_path}/{fname}', 'w') as file:
                        file.write(resp.text)
    print('Finished downloading.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        nargs='+',
        default=['en'],
        dest='langs',
        help='The languages to work on, separated by spaces. ' +
             'Possible options are en, de, nl, it, ' +
             'e.g. python multilang.py en nl it.'
    )
    args = parser.parse_args()

    langs: List[str] = args.langs
    if any([lang not in ['nl', 'de', 'it', 'en'] for lang in langs]):
        print('One of the language options was not recognized. ' +
              'Please choose from [en, nl, de, it]. Exiting...')
        exit()

    download_data(langs)
    taggers = [DecisionTreeTagger, SvmClassifier, TrigramTagger,
               HmmTagger, RnnTagger]
    os.makedirs('./output/', exist_ok=True)
    output = open('./output/multilang_output.txt', 'w')
    sys.stdout = output
    for lang in langs:
        print(f"LANGUAGE: {lang}")
        for tagger in taggers:
            t = tagger()
            print(f"CLASSIFIER: {type(t).__name__}")
            t.train(f'./data/{lang}/train.conll')
            t.accuracy(f'./data/{lang}/test.conll')
        print()  # Newline
    sys.stdout.close()
