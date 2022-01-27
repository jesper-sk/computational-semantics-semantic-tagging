import os
import argparse
import pandas as pd

import tensorflow as tf

from typing import List
from datetime import datetime
from util import classifier_options, download_data

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6611)])
        except RuntimeError as e:
            print(e)

    parser = argparse.ArgumentParser(
        description='Run any of the parsers on any of the languages in ' +
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

    if any([lang not in lang_options for lang in langs]):
        print('One of the language options was not recognized. ' +
              f'Please choose from {lang_options}. Exiting...')
        exit()

    if args.all_taggers:
        taggers = list(classifier_options.values())
    else:
        clf_names = list(classifier_options.keys())
        if any([clf not in clf_names for clf in clfs]):
            print(
                'One of the classifier options was not recognized. ' +
                f'Please choose from {clf_names}. Exiting...'
            )
            exit()
        taggers = [classifier_options[clf] for clf in clfs]

    print(
        f'Using classifiers: {[t.__name__ for t in taggers]} ' +
        'on languages: {langs}.'
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
