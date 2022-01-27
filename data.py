from typing import List, Tuple


def read_file(path):
    with open(path, 'r', encoding="utf8") as file:
        lines = file.readlines()

    labels = [
        "seg",
        "seg1",
        "sym",
        "sem",
        "cat",
        "sns",
        "rol",
    ]
    out = []
    for line in lines:
        if line.startswith('#') or line.startswith('\n'):
            continue

        tokens = line.split('\t')
        entry = {}
        for i, label in enumerate(labels):
            entry[label] = tokens[i].replace("\n", "")
        out.append(entry)

    return out


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

def nn_data(path):
    """
    Data formatted for neural networks
    """
    X = []
    y = []
    with open(path) as dataset:
        sentence = []
        tags = []
        for row in dataset:
            # Ignore comments
            if row.startswith('#'):
                continue
            # Start a new sentence with each new line
            if row.startswith('\n'):
                if len(sentence) > 0:
                    X.append(sentence)
                    y.append(tags)
                    sentence = []
                    tags = []
                continue
            fields = row.removesuffix('\n').split('\t')
            sentence.append(fields[1])
            tags.append(fields[3])
    return X, y

def test(path):
    with open(path, 'r') as file:
        print(file[1000:1020])


if __name__ == "__main__":
    from pprint import pprint
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-p', '--printc', type=int, default=20)
    parser.add_argument('-c',
                        '--classifier',
                        type=str.lower,
                        choices=['dt', 'tnt', 'def'])
    args = parser.parse_args()

    if args.classifier == 'tnt':
        d = tnt_data(args.path)
    elif args.classifier == 'dt':
        d = dt_data(args.path)
    else:
        d = read_file(args.path)

    pprint(d[:args.printc])
