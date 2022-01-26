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


def dt_data(path) -> Tuple[List[str], List[str]]:
    """Data formatted for the DecisionTree tagger
    """
    X = []  # data
    y = []  # target
    with open(path) as dataset:
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
    with open(path) as dataset:
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
