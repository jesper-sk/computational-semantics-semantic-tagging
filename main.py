import argparse
import sys
from baseline.DecisionTreeTagger import DecisionTreeTagger, \
                                        DecisionTreeTaggerOptions
from baseline.SvmTagger import SvmClassifier
from baseline.TrigramTagger import TrigramTagger
from baseline.HiddenMarkovModel import HmmTagger
from tagger.EnsembleTagger import EnsembleTagger


def main(args):
    if args.classifier == 'dt':
        opts = DecisionTreeTaggerOptions()
        opts.load_pretrained = not args.force_train
        tagger = DecisionTreeTagger(opts)
    elif args.classifier == 'tnt':
        tagger = TrigramTagger()
    elif args.classifier == 'svm':
        tagger = SvmClassifier()
    elif args.classifier == 'hmm':
        tagger = HmmTagger()
    elif args.classifier == 'ens':
        tagger = EnsembleTagger()

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
                            HMM: Hidden Markov Model
                            ENS: Ensemble (DT + SVM)''',
                        choices=['dt', 'tnt', 'svm', 'hmm', 'ens'],
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
