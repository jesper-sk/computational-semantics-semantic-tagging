def read_file(path):
    with open(path, 'r') as file:
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
        for i,label in enumerate(labels):
            entry[label] = tokens[i].replace("\n", "")
        out.append(entry)

    return out

def dt_data(path):
    X = [] # data
    y = [] # target
    with open(path) as dataset:
        for row in dataset:
            if row.startswith(('#', '\n')):
                continue
            fields = row.removesuffix('\n').split('\t')
            X.append(fields[1]) # word
            y.append(fields[3]) # semtag

    return X, y

if __name__ == "__main__":
    from pprint import pprint
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-p', '--printc', type=int, default=20)
    args = parser.parse_args()

    d = read_file(args.path)
    pprint(d[:args.printc])
