############
# Imports
############

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.layers.wrappers import TimeDistributed
from gensim.models import KeyedVectors
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import numpy as np
from baseline.WordEmbeddingClassifier import WordEmbeddingClassifier
from data import nn_data, dt_data
import fasttext

############
# RNN Class
############


class RNNTagger(WordEmbeddingClassifier):
    """
    A semantic tagger using a recurrant neural network
    """
    def __init__(self, lang: str = 'en'):
        super().__init__(lang)
        self.__model = None
        self.__max_vec_length = None
        self.__y_num_classes = None

    # def __get_word_embedding(self):
    #     path_name = "models/GoogleNews-vectors-negative300.bin"
    #     if os.path.exists("models/GoogleNews-vectors-negative300.bin"):
    #         print("Using word2vec word embeddings...")
    #         return path_name
    #     else:
    #         print(
    #             "Word embeddings not found; running without embedding weights..."
    #         )

    def __prepare_training_data(self, input_data):
        X, y = dt_data(input_data)
        X_vec, X_index, X_tokenizer = self.__vectorize(X)
        y_vec, y_index, y_tokenizer = self.__vectorize(y)
        self.__max_vec_length = max([len(i) for i in X_vec])
        X_pad = self.__pad_sequence(X_vec)
        y_pad = self.__pad_sequence(y_vec)
        X, y = X_pad, y_pad
        y = to_categorical(y, num_classes=75)
        self.__y_num_classes = y.shape[2]
        return X, y, X_index

    def __prepare_test_data(self, input_data):
        X, y = nn_data(input_data)
        X_vec, X_index, X_tokenizer = self.__vectorize(X)
        y_vec, y_index, y_tokenizer = self.__vectorize(y)
        X_pad = self.__pad_sequence(X_vec)
        y_pad = self.__pad_sequence(y_vec)
        X, y = X_pad, y_pad
        y = to_categorical(y, num_classes=75)
        return X, y, X_tokenizer, y_tokenizer

    def __vectorize(self, text):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text)
        return tokenizer.texts_to_sequences(
            text), tokenizer.word_index, tokenizer

    def __pad_sequence(self, vector):
        return pad_sequences(vector,
                             maxlen=self.__max_vec_length,
                             padding="pre",
                             truncating="post")

    def __word_embed(self, word_index, path=None):
        # if path == None:
        #     path = "models/GoogleNews-vectors-negative300.bin"
        # word2vec = KeyedVectors.load_word2vec_format(path, binary=True)
        EMBEDDING_SIZE = 300
        VOCABULARY_SIZE = len(word_index) + 1
        embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))
        word2id = word_index
        for word, index in word2id.items():
            try:
                embedding_weights[index, :] = self.word_embedding[word]
            except KeyError:
                pass
        return embedding_weights

    def train(self, input_data):
        X, y, X_index = self.__prepare_training_data(input_data)
        embedding_weights = self.__word_embed(X_index)

        # Setting constants for RNN training
        EMBEDDING_SIZE = 300
        VOCABULARY_SIZE = len(X_index) + 1
        MAX_SEQ_LENGTH = max([len(i) for i in X])
        NUM_CLASSES = self.__y_num_classes

        # RNN Instantion
        rnn = Sequential()
        rnn.add(
            Embedding(input_dim=VOCABULARY_SIZE,
                      output_dim=EMBEDDING_SIZE,
                      input_length=MAX_SEQ_LENGTH,
                      weights=[embedding_weights],
                      trainable=True))
        rnn.add(SimpleRNN(64, return_sequences=True))
        rnn.add(TimeDistributed(Dense(NUM_CLASSES, activation="softmax")))
        rnn.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["acc"])

        # Train RNN model
        rnn.fit(X, y, batch_size=128, epochs=10)
        self.__model = rnn

    def accuracy(self, input_data):
        if self.__model is None:
            print("Please train the RNN first.")
            return None
        else:
            X_test, y_test, _, _ = self.__prepare_test_data(input_data)
            loss, accuracy = self.__model.evaluate(X_test, y_test)
            print(f"Acc: {accuracy * 100:.02f}")
            return accuracy * 100

    def classify(self, input_data):
        if self.__model is None:
            print("Please train the RNN first.")
        else:
            X_test, y_test, X_tokenizer, y_tokenizer = self.__prepare_test_data(
                input_data)
            y_new = [[np.argmax(i) for i in j]
                     for j in self.__model.predict(X_test)]
            X_texts = X_tokenizer.sequences_to_texts(X_test)
            y_texts = y_tokenizer.sequences_to_texts(y_new)
            return X_texts, y_texts


#######
# Test
#######


def test():
    rnn = RNNTagger()
    input_data = "train.conll"
    test_data = "test.conll"
    rnn.train(input_data)
    rnn.accuracy(test_data)
    sentences, tags = rnn.classify(test_data)
    for i in range(10):
        print("-------------")
        print(i, ":")
        print(sentences[i])
        print(tags[i])
