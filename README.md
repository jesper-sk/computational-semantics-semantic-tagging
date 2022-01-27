# Semantic Tagging using Machine Learning

Computational Semantics, University of Groningen, block 1b 2021–2022

## Getting started

Install requirements:

```bash
pip install -r requirements.txt
```

## Classifying new data

```txt
$ python classify.py --help
usage: classify.py [-h] -l {en,nl,de,it} [-c {dt,svm,tnt,hmm,ens,rnn,lstm,gru,bilstm,birnn}] data_path

Find semantic tags for the Parallel Meaning Bank

positional arguments:
  data_path             Path to the data to classify, in the same format as the data used in 
                        Rik van Noorts GitHub repository. When not provided, a model will be 
                        trained without classifying anything.

optional arguments:
  -h, --help            show this help message and exit
  -l {en,nl,de,it}, --language {en,nl,de,it}
                        The language the data to classify is in.
  -c {dt,svm,tnt,hmm,ens,rnn,lstm,gru,bilstm,birnn}, --classifier {dt,svm,tnt,hmm,ens,rnn,lstm,gru,bilstm,birnn}
                        Name   | Description
                        -------|----------------------------------------------------
                        DT     | Decision tree 
                        SVM    | Support Vector Machine 
                        TNT    | Trigrams'n'Tags 
                        HMM    | Hidden Markov Model 
                        RNN    | Recurrent Neural Network 
                        ENS    | Ensemble (DT + SVM) 
                        LSTM   | Long Short Term Memory Neural Network 
                        GRU    | Gated Recurrent Unit Neural Network 
                        BiRNN  | Bidirectional Recurrent Neural Network 
                        BiLSTM | Bidirectional Long Short Term Memory Neural Network
```

For example, if you have a data set `data.conll` which is in Italian, and you want to use the RNN, you run the script as
follows:

```bash
python classify.py data.conll -l it -c rnn
```

## Testing multiple classifiers (on multiple data sets)

To run multiple classifiers in this repository on the available datasets in the PMB, you can use
`test.py` as follows:

```bash
$ python test.py --help                                                                                                                                                                                                 main*
usage: test.py [-h] [-c CLASSIFIERS [CLASSIFIERS ...]] [-a] langs [langs ...]

Run any of the parsers on any of the languages in the PMB data set.

positional arguments:
  langs                 The languages to work on, separated by spaces. 
                        Possible options are [en, de, nl, it], e.g. python multilang.py en nl it.

optional arguments:
  -h, --help            show this help message and exit
  -c CLASSIFIERS [CLASSIFIERS ...], --classifiers CLASSIFIERS [CLASSIFIERS ...]
                        The classifiers to use, separated by spaces. 
                        Possible options are [dt, tnt, svm, hmm, ens, rnn, lstm, gru, birnn, bilstm]
  -a, --all-taggers     Use all taggers. Ignores options supplied to the-c / --classifier argument.
```

This will download the data sets and run the models.

**NOTE**: This *will* take a long time and takes up quite a bit of disk space. Several models depend on
language-specific word embeddings, which each take up about 7.24GB. Thus, if you run the classifiers for all languages,
you need to have about 29GB of disk space available. Running a classifier on a single language can take up to half an
hour, depending on the hardware used.
