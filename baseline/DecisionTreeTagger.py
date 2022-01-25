from datetime import datetime
from typing import List
from dataclasses import dataclass
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from baseline.WordEmbeddingClassifier import WordEmbeddingClassifier
import os
import pickle
import numpy as np


@dataclass
class DecisionTreeTaggerOptions:
    """Options for the decision tree tagger.
    """

    # Whether to use grid search to find best parameters
    use_grid_search: bool = True
    # Whether to load a pretrained model if one is available
    load_pretrained = True


class DecisionTreeTagger(WordEmbeddingClassifier):
    """A semantic tagger using decision trees
    """
    def __init__(self, options: DecisionTreeTaggerOptions = None, lang: str = 'en') -> None:
        super().__init__(lang=lang)
        self.__options = options or DecisionTreeTaggerOptions(
        )  # take defaults if no options given
        self.__model = None

    def load_model(self):
        """Load a pretrained model"""
        if os.path.exists(f'./models/dt_model_{self.lang}.pkl'):
            with open(f'./models/dt_model_{self.lang}.pkl', 'rb') as model_pickle:
                self.__model = pickle.load(model_pickle)
            return True
        return False

    def train(self, input_data: str) -> float:
        """Train a decision tree tagger on some data set
        """
        if self.__options.load_pretrained:
            if self.load_model():
                print("""Found pretrained decision tree model.
                Skipping training (use --force-train to force training).""")
                return
            else:
                print(
                    "No pretrained decision tree model found, training now...")

        _, data_in, data_out = self.prepare_data(input_data)
        kf = KFold(10, shuffle=True)
        dtree = tree.DecisionTreeClassifier()
        for k, (train, test) in enumerate(kf.split(data_in, data_out)):
            dtree.fit(data_in[train], data_out[train])
            print(
                "[fold {0}] score: {1:.5f}".format(
                    k, dtree.score(data_in[test], data_out[test])
                )
            )
        self.__model = dtree

        # Save model to file
        os.makedirs('./models/', exist_ok=True)
        with open(f'./models/dt_model_{self.lang}.pkl', 'wb') as model_pickle:
            pickle.dump(self.__model, model_pickle)

    def accuracy(self, input_path) -> float:
        """Evaluates the accuracy of the model on a (tagged) test set"""
        if self.__model is None:
            print("No model available. Please train the model first.")
            return None
        else:
            _, data_vectors, true_tags = self.prepare_data(input_path)
            predictions = self.__model.predict(data_vectors)
            acc = accuracy_score(true_tags, predictions)
            input_filename = os.path.basename(input_path)
            print(f"""This model has an accuracy of {acc * 100:.2f}% on
            {input_filename}.""")
            
            return acc * 100

    def classify(self, input_path) -> List:
        """Classify new data after training. Expects a list of words as input.
        Saves the output to a file.
        """
        if self.__model is None:
            print("No model available. Please train the model first.")
        else:
            words, data_vectors, _ = self.prepare_data(input_path)
            predictions = self.__model.predict(data_vectors)
            input_filename = os.path.basename(input_path)
            os.makedirs('./output/', exist_ok=True)
            file_name = f'./output/dt_{input_filename}_{datetime.now():%Y-%m-%d_%H%M}.tsv'
            with open(file_name, 'w') as f:
                f.write("word\tpredicted tag\n")
                for (d_input, d_output) in zip(words, predictions):
                    f.write(d_input + '\t' + d_output + '\n')

            print(f'Output written to file {file_name}')
