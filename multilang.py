import os
import requests
import argparse
import pandas as pd
from typing import List
from datetime import datetime
from baseline.DecisionTreeTagger import DecisionTreeTagger
from baseline.SvmTagger import SvmClassifier
from baseline.TrigramTagger import TrigramTagger
from baseline.HiddenMarkovModel import HmmTagger
from tagger.EnsembleTagger import EnsembleTagger
from tagger.RNNTagger import RNNTagger
from tagger.BiRNNTagger import BiRNNTagger
from tagger.LSTMTagger import LSTMTagger
from tagger.BiLSTMTagger import BiLSTMTagger
from tagger.GRUTagger import GRUTagger


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
        for fname in ['train.conll', 'test.conll', 'dev.conll', 'eval.conll']:
            url = f'{base_url}/{lang}/gold/{fname}'
            os.makedirs(save_path, exist_ok=True)

            if os.path.exists(f'{save_path}/{fname}') and not force:
                print(f'{fname : <11} for {lang} found, skipping download')
            else:
                resp = requests.get(url)
                if not resp.ok:
                    print(f'File {fname} does not exist for {lang}. skipping...')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run any of the parsers on any of the languages in' +
        'the PMB data set.')

    parser.add_argument(
        nargs='+',
        default=['en'],
        dest='langs',
        help='The languages to work on, separated by spaces. ' +
        'Possible options are [en, de, nl, it], ' +
        'e.g. python multilang.py en nl it.')
    parser.add_argument(
        '-c',
        '--classifiers',
        nargs='+',
        default=['tnt'],
        help='The classifiers to use. Possible options are [dt, tnt, svm, ' +
        'hmm, ens, rnn, lstm, gru, birnn, bilstm]')
    parser.add_argument(
        '-a',
        '--all-taggers',
        help='Use all taggers. Ignores options supplied to the' +
        '-c / --classifier argument.',
        action='store_true')
    args = parser.parse_args()
    print(args)

    langs: List[str] = args.langs
    clfs: List[str] = args.classifiers

    lang_options = ['nl', 'de', 'it', 'en']
    classifier_options = {
        'dt': DecisionTreeTagger,
        'svm': SvmClassifier,
        'tnt': TrigramTagger,
        'hmm': HmmTagger,
        'ens': EnsembleTagger,
        'rnn': RNNTagger,
        'lstm': LSTMTagger,
        'gru': GRUTagger,
        'bilstm': BiLSTMTagger,
        'birnn': BiRNNTagger,
    }

    if any([lang not in lang_options for lang in langs]):
        print('One of the language options was not recognized. ' +
              f'Please choose from {lang_options}. Exiting...')
        exit()

    if args.all_taggers:
        taggers = list(classifier_options.values())
    else:
        if any([clf not in list(classifier_options.keys()) for clf in clfs]):
            print(
                'One of the classifier options was not recognized. ' +
                f'Please choose from {list(classifier_options.keys())}. Exiting...'
            )
            exit()
        taggers = [classifier_options[clf] for clf in clfs]

    print(
        f'Using classifiers: {[t.__name__ for t in taggers]} on languages: {langs}.'
    )

    download_data(langs)
    os.makedirs('./output/', exist_ok=True)
    df = pd.DataFrame()
    for lang in langs:
        for tagger in taggers:
            print(f'== Current model: {lang} / {tagger.__name__} ==')
            t = tagger(lang=lang)
            t.train(f'./data/{lang}/all.conll')
            acc = t.accuracy(f'./data/{lang}/test.conll')
            df.at[lang, tagger.__name__] = acc

    fname = f'./output/multilang_output_{datetime.now():%Y-%m-%d_%H%M}.csv'
    df.to_csv(fname)
    print(df)
    print(f"Done. Output written to {fname}.")
