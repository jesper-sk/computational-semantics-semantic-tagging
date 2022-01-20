import os
import pickle
from typing import List
from dataclasses import dataclass
from datetime import datetime
from sklearn import pipeline, svm
from baseline.WordEmbeddingClassifier import WordEmbeddingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


@dataclass
class EnsembleTaggerOptions:
    """Options for the ensemble tagger.
    """
    # Whether to load a pretrained model if one is available
    load_pretrained = True

class EnsembleTagger(WordEmbeddingClassifier):
    """An ensemble tagger. NOTE: does not work yet.
    """
    def __init__(self, options: EnsembleTaggerOptions = None) -> None:
        super().__init__()
        self.__options = options or EnsembleTaggerOptions()
        self.__model = None

    def train(self, input_data: str) -> None:
        """Train an ensemble of classifiers.
        """
        if self.__options.load_pretrained:
            if self.load_model():
                print("""Found pretrained svm model. Skipping training
                (use --force-train to force training).""")
                return
            else:
                print("No pretrained svm model found, training now...")
        # Prepare data
        _, data_in, data_out = self.prepare_data(input_data)
        in_train, in_test, out_train, out_test = train_test_split(
            data_in, data_out)

        # Prepare some models
        dtc = DecisionTreeClassifier()
        svc = pipeline.make_pipeline(StandardScaler(),
                                     svm.LinearSVC(dual=False, C=10))

        # Create ensemble model
        vcl = VotingClassifier(estimators=[('dt', dtc), ('svm', svc)])
        vcl.fit(X=in_train, y=out_train)
        self.__model = vcl

        # Check accuracy on test set
        out_predicted = self.__model.predict(in_test)
        acc = accuracy_score(out_test, out_predicted)
        print(f"This model has a training accuracy of {acc * 100:.2f}%.")

        # Save model to file
        os.makedirs('./models/', exist_ok=True)
        with open('./models/ensemble_model.pkl', 'wb') as model_pickle:
            pickle.dump(self.__model, model_pickle)

    def load_model(self):
        if os.path.exists('./models/ensemble_model.pkl'):
            with open('./models/ensemble_model.pkl', 'rb') as model_pickle:
                self.__model = pickle.load(model_pickle)
            return True
        else:
            return False

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