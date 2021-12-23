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

if __name__ == "__main__":
    from pprint import pprint
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('path', required=True)
    parser.add_argument('-p', '--printc', type=int, default=20)
    args = parser.parse_args()

    d = read_file(args.path)
    pprint(d[:args.printc])
