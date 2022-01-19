from datetime import datetime
from typing import List
from dataclasses import dataclass
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from gensim.models.fasttext import FastText
from gensim.test.utils import datapath

import data
import os
import pickle

@dataclass
class SvmClassifierOptions:
    """Options for the svm tagger.
    """
    # which SVm classifier from sklearn to use
    # svm: svm.SVC(C=10, kernel='linear')
    svm = svm.LinearSVC(dual=False, C=10)
    # Whether to perform n-fold cross-validation, and if so, for what value of n
    n_fold_cv: int = None
    # Whether to use grid search to find best parameters
    use_grid_search: bool = False
    # The fraction of the data to use as training data
    train_split: float = 0.2
    # Whether to load a pretrained model if one is available
    load_pretrained = True


class SvmClassifier:
    """A semantic tagger using a support-vector machine.
    """
    def __init__(self, options: SvmClassifierOptions = None) -> None:
        self.__options = options or SvmClassifierOptions() # take defaults if no options given
        self.__model = None
        self.__word_embedding = self.__get_word_embedding()

    def __get_word_embedding(self) -> FastText:
        path_name = './models/fasttext.mod'
        if os.path.exists(path_name):
            print("Found pre-trained word embedding")
            return FastText.load(path_name)
        else:
            print("No pre-trained word embedding found, training now...")
            # Training on Lee corpus
            corpus_file = datapath('lee_background.cor')
            model = FastText(vector_size=100)
            model.build_vocab(corpus_file=corpus_file)
            model.train(
                corpus_file=corpus_file, epochs=model.epochs,
                total_examples=model.corpus_count, total_words=model.corpus_total_words
            )
            print("Training finished. Saving model for next use...")
            os.makedirs('./models/', exist_ok=True)
            model.save(path_name, separately=[])
            return model


    def __prepare_data(self, input_data):
        in_raw, out = data.dt_data(input_data)
        # Prepare the data by transforming words to vectors
        in_vectors = self.__word_embedding.wv[in_raw]
        return in_raw, in_vectors, out


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
                print("Found pretrained svm model. Skipping training (use --force-train to force training).")
                return
            else:
                print("No pretrained svm model found, training now...")

        _, data_in, data_out = self.__prepare_data(input_data)
        in_train, in_test, out_train, out_test = train_test_split(data_in, data_out)



        if self.__options.use_grid_search:
            params = {
                # TODO: params 
                'C' : [0.1, 1, 10, 100, 1000],
            }
            clf = make_pipeline(
                StandardScaler(), 
                GridSearchCV(
                    self.__options.svm, 
                    params, 
                    n_jobs=4,
                    refit=True
                    )
                )
        else:
            clf = make_pipeline(
                StandardScaler(), 
                self.__options.svm
                )

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
            words, data_vectors, true_tags = self.__prepare_data(input_path)
            predictions = self.__model.predict(data_vectors)
            acc = accuracy_score(true_tags, predictions)
            input_filename = os.path.basename(input_path)
            print(f"This model has an accuracy of {acc * 100:02}% on {input_filename}.")


    def classify(self, input_path) -> List:
        """Classify new data after training. Expects a list of words as input.
        Saves the output to a file.
        """
        if self.__model is None:
            print("No model available. Please train the model first.")
        else:
            words, data_vectors, _ = self.__prepare_data(input_path)
            predictions = self.__model.predict(data_vectors)
            input_filename = os.path.basename(input_path)
            os.makedirs('./output/', exist_ok=True)
            file_name = f'./output/svm_{input_filename}_{datetime.now():%Y-%m-%d_%H%M}.tsv'
            with open(file_name, 'w') as f:
                f.write("word\tpredicted tag\n")
                for (d_input, d_output) in zip(words, predictions):
                    f.write(d_input + '\t' + d_output + '\n')
            
            print(f'Output written to file {file_name}')
