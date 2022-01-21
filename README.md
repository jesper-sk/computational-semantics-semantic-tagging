# Semantic Tagging using Machine Learning

Computational Semantics, University of Groningen, block 1b 2021–2022

## Getting started

1. Install requirements:

    ```bash
    pip install -r requirements.txt
    ```

2. Download `train.conll` and `test.conll` files from [here](https://github.com/RikVN/DRS_parsing/tree/master/parsing/layer_data/4.0.0/en/gold) and place them in the directory

## Running the program

```txt
$ python main.py --help
usage: main.py [-h] [-t TRAINING_DATA] [-c {dt,tnt}] [-d DATA] [-f]

Find semantic tags for the Parallel Meaning Bank

optional arguments:
  -h, --help            show this help message and exit
  -t TRAINING_DATA, --training_data TRAINING_DATA
                        The file to use as training data.
  -c {dt,tnt}, --classifier {dt,tnt}
                        The classifier to use. DT: decision tree; TNT: Trigrams'n'tags
  -d DATA, --data DATA  Data to classify
  -f, --force-train     Train a new model even if one exists already (default: False)
```

For example, if you want to train a decision tree classifier using the training data in `train.conll`:

```bash
python main.py -t train.conll -c dt
```

Or if you want to classify new data using a pre-trained decision tree classifier:

```bash
python main.py -d test.conll -c dt
```

## Testing

To run all the classifiers in this repository on all datasets, you can use
`multilang.py` as follows:

```bash
usage: multilang.py [-h] [-c CLASSIFIERS [CLASSIFIERS ...]] langs [langs ...]

positional arguments:
  langs                 The languages to work on, separated by spaces. Possible options are [en, de, nl, it], e.g. python multilang.py en nl it.

optional arguments:
  -h, --help            show this help message and exit
  -c CLASSIFIERS [CLASSIFIERS ...], --classifiers CLASSIFIERS [CLASSIFIERS ...]
                        The classifiers to use. Possible options are[dt, tnt, svm, hmm, ens, rnn]
```

This will download the data sets and run the models.

**NOTE*: This *will* take a long time and take up quite a bit of disk space. The
decision tree model and SVM depend on language-specific word embeddings, which
each take up about 7.24GB. Thus, if you run the classifiers for all languages,
you need to have about 29GB of disk space available.
