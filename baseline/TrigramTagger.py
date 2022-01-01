from dataclasses import dataclass
from datetime import datetime
import nltk
import os
import data

@dataclass
class TrigramTaggerOptions():
    pass

class TrigramTagger:
    def __init__(self, options: TrigramTaggerOptions = None) -> None:
        self.__options = options or TrigramTaggerOptions()
        self.__model = None
    
    def train(self, input_path: str) -> None:
        training_data = data.tnt_data(input_path)
        tagger = nltk.tag.TnT()
        tagger.train(training_data)
        self.__model = tagger

    def accuracy(self, input_path):
        """Evaluates the accuracy of the model on a (tagged) test set"""
        if self.__model is None:
            print("No model available. Please train the model first.")
        else:
            input_filename = os.path.basename(input_path)
            input_data = data.tnt_data(input_path)
            acc = self.__model.accuracy(input_data)
            print(f"This model has an accuracy of {acc * 100:.2f}% on {input_filename}.")

    def classify(self, input_path):
        """Classify new data after training. Expects a list of words as input.
        Saves the output to a file.
        """
        if self.__model is None:
            print("No model available. Please train the model first.")
        else:
            # For now, just take as input the original format. For actual use,
            # this function should take as input a list instead of a path.
            input_data = data.tnt_data(input_path)
            input_data_notags = [ [ word for (word, _) in sentence ] for sentence in input_data ]
            predictions = self.__model.tagdata(input_data_notags)
            input_filename = os.path.basename(input_path)
            os.makedirs('./output/', exist_ok=True)
            file_name = f'./output/tnt_{input_filename}_{datetime.now():%Y-%m-%d_%H:%M}.tsv'
            with open(file_name, 'w') as f:
                f.write("word\tpredicted tag\n")
                for sentence in predictions:
                    for (d_input, d_output) in sentence:
                        f.write(d_input + '\t' + d_output + '\n')

            print(f'Output written to file {file_name}')
