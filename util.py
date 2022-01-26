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
