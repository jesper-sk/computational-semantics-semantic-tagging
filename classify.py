import argparse
import sys
from download_data import download_data
from util import classifier_options


def main(args):
    download_data([args.language])
    classifier = classifier_options[args.classifier]()
    classifier.train(f'./data/{args.language}/all.conll')

    if args.data_path is not None:
        classifier.classify(args.data_path)
        classifier.accuracy(args.data_path)

    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find semantic tags for the Parallel Meaning Bank",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        help='Path to the data to classify, in the same format as the data ' +
        'used in Rik van Noorts GitHub repository. When not provided, a ' +
        'model will be trained without classifying anything.',
        type=str,
        default=None,
        dest='data_path')
    parser.add_argument('-l',
                        '--language',
                        help='The language the data to classify is in.',
                        type=str,
                        choices=['en', 'nl', 'de', 'it'],
                        required=True)
    parser.add_argument(
        '-c',
        '--classifier',
        help='Name   | Description\n'
        '-------|----------------------------------------------------\n'
        'DT     | Decision tree \n' + 'SVM    | Support Vector Machine \n' +
        'TNT    | Trigrams\'n\'Tags \n' + 'HMM    | Hidden Markov Model \n' +
        'RNN    | Recurrent Neural Network \n' +
        'ENS    | Ensemble (DT + SVM) \n' +
        'LSTM   | Long Short Term Memory Neural Network \n' +
        'GRU    | Gated Recurrent Unit Neural Network \n' +
        'BiRNN  | Bidirectional Recurrent Neural Network \n' +
        'BiLSTM | Bidirectional Long Short Term Memory Neural Network',
        choices=list(classifier_options.keys()),
        type=str.lower)

    main(parser.parse_args())
