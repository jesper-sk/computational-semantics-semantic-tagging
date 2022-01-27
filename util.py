from typing import List, Tuple
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
import requests
import os

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
        with open(f'{save_path}/all.conll', 'w', encoding="utf-8") as all_txt:
            with open(f'{save_path}/train.conll', 'r', encoding="utf-8") as train_txt,\
                 open(f'{save_path}/dev.conll', 'r', encoding="utf-8") as test_txt:
                all_txt.write(train_txt.read())
                all_txt.write(test_txt.read())
            # Try to attach eval.conll if it exists (doesn't for all languages)
            try:
                with open(f'{save_path}/eval.conll', 'r') as eval_txt:
                    all_txt.write(eval_txt.read())
            except IOError:
                continue  # Just ignore error if eval.conll doesn't exist

    print('Finished downloading.')


def dt_data(path) -> Tuple[List[str], List[str]]:
    """Data formatted for the DecisionTree tagger
    """
    X = []  # data
    y = []  # target
    with open(path, encoding="utf-8") as dataset:
        for row in dataset:
            if row.startswith(('#', '\n')):
                continue
            fields = row.removesuffix('\n').split('\t')
            X.append(fields[1])  # word
            y.append(fields[3])  # semtag

    return X, y


def tnt_data(path) -> List[List[Tuple[str, str]]]:
    """Data formatted for the TnT tagger.
    Returns a list of tagged sentences [[(word, tag)]]"""
    data = []
    with open(path, encoding="utf-8") as dataset:
        sentence = []
        for row in dataset:
            # Ignore comments
            if row.startswith('#'):
                continue
            # Start a new sentence when encountering a newline
            if row.startswith('\n'):
                # Entries are separated by multiple newlines
                if len(sentence) > 0:
                    data.append(sentence)
                    sentence = []
                continue

            fields = row.removesuffix('\n').split('\t')
            sentence.append((fields[1], fields[3]))

    return data
