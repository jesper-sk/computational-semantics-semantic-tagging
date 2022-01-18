from datetime import datetime
from typing import List
from dataclasses import dataclass
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from baseline.WordEmbeddingClassifier import WordEmbeddingClassifier

import os
import pickle


@dataclass
class SvmClassifierOptions:
    """Options for the svm tagger.
    """
    # The kernel function for the svm
    kernel: str = "linear"
    # Whether to use grid search to find best parameters
    use_grid_search: bool = False
    # Whether to load a pretrained model if one is available
    load_pretrained = True


class SvmClassifier(WordEmbeddingClassifier):
    """A semantic tagger using a support-vector machine.
    """
    def __init__(self, options: SvmClassifierOptions = None) -> None:
        super().__init__()
        # take defaults if no options given
        self.__options = options or SvmClassifierOptions()
        self.__model = None

    def load_model(self):
        if os.path.exists('./models/svm_model.pkl'):
            with open('./models/svm_model.pkl', 'rb') as model_pickle:
                self.__model = pickle.load(model_pickle)
            return True
        else:
            return False

    def train(self, input_data: str) -> None:
        """Train a svm tagger on some data set
        """
        if self.__options.load_pretrained:
            if self.load_model():
                print("""Found pretrained svm model. Skipping training
                (use --force-train to force training).""")
                return
            else:
                print("No pretrained svm model found, training now...")

        _, data_in, data_out = self.prepare_data(input_data)
        in_train, in_test, out_train, out_test = train_test_split(
            data_in, data_out)

        if self.__options.use_grid_search:
            params = {
                # TODO: params
                'C': [0.1, 1, 10, 100, 1000],
            }
            clf = make_pipeline(
                StandardScaler(),
                GridSearchCV(svm.SVC(kernel=self.__options.kernel),
                             params,
                             n_jobs=4,
                             refit=True))
        else:
            clf = make_pipeline(StandardScaler(),
                                svm.SVC(kernel=self.__options.kernel))

        clf.fit(X=in_train, y=out_train)
        self.__model = clf

        # Check accuracy on test set
        out_predicted = self.__model.predict(in_test)
        acc = accuracy_score(out_test, out_predicted)
        print(f"This model has a training accuracy of {acc * 100:.2f}%.")

        # Save model to file
        os.makedirs('./models/', exist_ok=True)
        with open('./models/svm_model.pkl', 'wb') as model_pickle:
            pickle.dump(self.__model, model_pickle)

    def accuracy(self, input_path):
        """Evaluates the accuracy of the model on a (tagged) test set"""
        if self.__model is None:
            print("No model available. Please train the model first.")
        else:
            _, data_vectors, true_tags = self.prepare_data(input_path)
            predictions = self.__model.predict(data_vectors)
            acc = accuracy_score(true_tags, predictions)
            input_filename = os.path.basename(input_path)
            print(f"""This model has an accuracy of {acc * 100:02}% on
                {input_filename}.""")

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
            file_name = f"""./output/svm_{input_filename}_
            {datetime.now():%Y-%m-%d_%H%M}.tsv"""
            with open(file_name, 'w') as f:
                f.write("word\tpredicted tag\n")
                for (d_input, d_output) in zip(words, predictions):
                    f.write(d_input + '\t' + d_output + '\n')

            print(f'Output written to file {file_name}')
