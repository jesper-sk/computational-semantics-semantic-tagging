import argparse
import os

from matplotlib import pyplot as plt

from data import read_file


def count(dbs):
    langs = list(dbs.keys())
    all_sems = []
    for lang in langs:
        print(lang)
        lang_sems = []
        for file, data in dbs[lang].items():
            sems = [ token['sem'] for token in data ]
            lang_sems += sems
            all_sems += sems
            print(f'\t{file}\t--> {len(data)} / {len(set(sems))}')
        print(f"n.sems --> f{len(set(lang_sems))}")

    plt.hist(all_sems)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')

    args = parser.parse_args()

    dbs = {}

    for name in os.listdir(args.dir):
        path = f"{args.dir}/{name}"
        if os.path.isdir(path):
            for file in os.listdir(path):
                if os.path.isfile(f"{path}/{file}"):
                    data = read_file(f"{path}/{file}")
                    file = file.split('.')[0]
                    if not dbs.get(name):
                        dbs[name] = { file : data }
                    else:
                        dbs[name][file] = data

    count(dbs)  

if __name__ == "__main__":
    main()