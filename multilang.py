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
from tagger.RNNTagger import RNNTagger
from tagger.EnsembleTagger import EnsembleTagger


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
        save_path = f'./data/{lang}'
        for fname in ['train.conll', 'test.conll', 'dev.conll']:
            url = f'{base_url}/{lang}/gold/{fname}'
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

        # Create merged train/dev set
        with open(f'{save_path}/train.conll', 'r') as train_txt:
            with open(f'{save_path}/dev.conll', 'r') as test_txt:
                with open(f'{save_path}/all.conll', 'w') as all_txt:
                    all_txt.write(train_txt.read())
                    all_txt.write(test_txt.read())

    print('Finished downloading.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        nargs='+',
        default=['en'],
        dest='langs',
        help='The languages to work on, separated by spaces. ' +
        'Possible options are [en, de, nl, it], ' +
        'e.g. python multilang.py en nl it.')
    parser.add_argument('-c',
                        '--classifiers',
                        nargs='+',
                        default=['tnt'],
                        help='The classifiers to use. Possible options are' +
                        '[dt, tnt, svm, hmm, ens, rnn]')
    args = parser.parse_args()

    langs: List[str] = args.langs
    clfs: List[str] = args.classifiers

    lang_options = ['nl', 'de', 'it', 'en']
    classifier_options = {
        'dt': DecisionTreeTagger,
        'svm': SvmClassifier,
        'tnt': TrigramTagger,
        'hmm': HmmTagger,
        'ens': EnsembleTagger,
        'rnn': RNNTagger
    }

    if any([lang not in lang_options for lang in langs]):
        print('One of the language options was not recognized. ' +
              f'Please choose from {lang_options}. Exiting...')
        exit()
    if any([clf not in list(classifier_options.keys()) for clf in clfs]):
        print('One of the classifier options was not recognized. ' +
              f'Please choose from {list(classifier_options.keys())}. Exiting...')
        exit()

    taggers = [classifier_options[clf] for clf in clfs]

    print(f'Using classifiers: {[t.__name__ for t in taggers]} on languages: {langs}.')

    download_data(langs)
    os.makedirs('./output/', exist_ok=True)
    with open('./output/multilang_output.txt', 'w') as output:
        for lang in langs:
            output.write(f"LANGUAGE: {lang}\n")
            for tagger in taggers:
                t = tagger(lang=lang)
                output.write(f"CLASSIFIER: {tagger.__name__}\n")
                t.train(f'./data/{lang}/all.conll')
                acc = t.accuracy(f'./data/{lang}/test.conll')
                output.write(f'ACCURACY: {acc}%\n')
            output.write('\n')
