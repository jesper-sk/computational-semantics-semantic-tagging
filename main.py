import argparse
import sys
from baseline.DecisionTreeTagger import DecisionTreeTagger, \
                                        DecisionTreeTaggerOptions
from baseline.SvmTagger import SvmClassifier, \
                               SvmClassifierOptions
from baseline.TrigramTagger import TrigramTagger
from baseline.HiddenMarkovModel import HmmTagger
from tagger.EnsembleTagger import EnsembleTagger
from tagger.RNNTagger import RNNTagger
from tagger.LSTMTagger import LSTMTagger
from tagger.GRUTagger import GRUTagger
from tagger.BiRNNTagger import BiRNNTagger
from tagger.BiLSTMTagger import BiLSTMTagger


def main(args):
    if args.classifier == 'dt':
        opts = DecisionTreeTaggerOptions()
        opts.load_pretrained = not args.force_train
        tagger = DecisionTreeTagger(opts)
    elif args.classifier == 'tnt':
        tagger = TrigramTagger()
    elif args.classifier == 'svm':
        opts = SvmClassifierOptions()
        opts.load_pretrained = not args.force_train
        tagger = SvmClassifier(opts)
    elif args.classifier == 'hmm':
        tagger = HmmTagger()
    elif args.classifier == 'ens':
        tagger = EnsembleTagger()
    elif args.classifier == 'rnn':
        tagger = RNNTagger()
    elif args.classifier == 'lstm':
        tagger = LSTMTagger()
    elif args.classifier == 'gru':
        tagger = GRUTagger()
    elif args.classifier == 'birnn':
        tagger = BiRNNTagger()
    elif args.classifier == 'bilstm':
        tagger = BiLSTMTagger()

    if args.training_data is not None:
        tagger.train(args.training_data)
    if args.data is not None:
        tagger.classify(args.data)
        tagger.accuracy(args.data)
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find semantic tags for the Parallel Meaning Bank")

    parser.add_argument('-t',
                        '--training_data',
                        help="The file to use as training data.",
                        type=str,
                        default=None)
    parser.add_argument('-c',
                        '--classifier',
                        help='''The classifier to use.
                            DT: decision tree;
                            SVM: Support Vector Machine;
                            TNT: Trigrams\'n\'Tags;
                            HMM: Hidden Markov Model;
                            RNN: Recurrent Neural Network;
                            ENS: Ensemble (DT + SVM);
                            LSTM: Long Short Term Memory Neural Network;
                            GRU: Gated Recurrant Unit Neural Network;
                            BiRNN: Bidirectional Recurrant Neural Network;
                            BiLSTM: Bidirectional Long Short Term Memory Neural Network''',
                        choices=['dt', 'tnt', 'svm', 'hmm', 'rnn', 'ens', 'lstm', 'gru', 'birnn', 'bilstm'],
                        type=str.lower)
    parser.add_argument('-d',
                        '--data',
                        help='Data to classify',
                        type=str,
                        default=None)
    parser.add_argument(
        '-f',
        '--force-train',
        help='Train a new model even if one exists already (default: False)',
        action='store_true')

    main(parser.parse_args())
