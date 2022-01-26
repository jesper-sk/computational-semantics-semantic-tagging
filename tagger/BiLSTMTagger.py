############
# Imports
############

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.layers.wrappers import TimeDistributed
from sklearn.model_selection import KFold
import numpy as np
from baseline.WordEmbeddingClassifier import WordEmbeddingClassifier
from util import dt_data

############
# BiLSTM Class
############


class BiLSTMTagger(WordEmbeddingClassifier):
    """
    A semantic tagger using a recurrant neural network
    """
    def __init__(self, lang: str = 'en'):
        super().__init__(lang)
        self.__model = None
        self.__max_vec_length = None
        self.__y_num_classes = None

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
        X, y = dt_data(input_data)
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

    def normal_train(self, input_data):
        X, y, X_index = self.__prepare_training_data(input_data)
        embedding_weights = self.__word_embed(X_index)

        # Setting constants for BiLSTM training
        EMBEDDING_SIZE = 300
        VOCABULARY_SIZE = len(X_index) + 1
        MAX_SEQ_LENGTH = max([len(i) for i in X])
        NUM_CLASSES = self.__y_num_classes

        # BiLSTM Instantion
        blstm = Sequential()
        blstm.add(
            Embedding(input_dim=VOCABULARY_SIZE,
                      output_dim=EMBEDDING_SIZE,
                      input_length=MAX_SEQ_LENGTH,
                      weights=[embedding_weights],
                      trainable=True))
        blstm.add(
            Bidirectional(
                LSTM(64,
                     kernel_regularizer='l2',
                     activity_regularizer='l2',
                     return_sequences=True)))
        blstm.add(
            TimeDistributed(
                Dense(NUM_CLASSES,
                      activation="softmax",
                      kernel_regularizer='l2',
                      activity_regularizer='l2')))
        blstm.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["acc"])

        # Train BiLSTM model
        blstm.fit(X, y, batch_size=128, epochs=10)
        self.__model = blstm

    def train(self, input_data):
        X, y, X_index = self.__prepare_training_data(input_data)
        embedding_weights = self.__word_embed(X_index)

        # Setting constants for BiLSTM training
        EMBEDDING_SIZE = 300
        VOCABULARY_SIZE = len(X_index) + 1
        MAX_SEQ_LENGTH = max([len(i) for i in X])
        NUM_CLASSES = self.__y_num_classes

        # BiLSTM Instantion
        blstm = Sequential()
        blstm.add(
            Embedding(input_dim=VOCABULARY_SIZE,
                      output_dim=EMBEDDING_SIZE,
                      input_length=MAX_SEQ_LENGTH,
                      weights=[embedding_weights],
                      trainable=True))
        blstm.add(
            Bidirectional(
                LSTM(64,
                     kernel_regularizer='l2',
                     activity_regularizer='l2',
                     return_sequences=True)))
        blstm.add(
            TimeDistributed(
                Dense(NUM_CLASSES,
                      activation="softmax",
                      kernel_regularizer='l2',
                      activity_regularizer='l2')))
        blstm.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["acc"])

        # Train BiLSTM model
        kf = KFold(10, shuffle=True)
        for k, (train, test) in enumerate(kf.split(X, y)):
            blstm.fit(X[train], y[train], batch_size=128, epochs=10)
            loss, accuracy = blstm.evaluate(X[test], y[test], verbose=False)
            print(f"[fold{k}] score : {accuracy:.5f}")
        self.__model = blstm

    def accuracy(self, input_data):
        if self.__model is None:
            print("Please train the BiLSTM first.")
            return None
        else:
            X_test, y_test, _, _ = self.__prepare_test_data(input_data)
            loss, accuracy = self.__model.evaluate(X_test, y_test)
            print(f"Acc: {accuracy * 100:.02f}")
            return accuracy * 100

    def classify(self, input_data):
        if self.__model is None:
            print("Please train the BiLSTM first.")
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


def printtags():
    for i in range(10):
        print("-------------")
        print(i, ":")
        print(sentences[i])
        print(tags[i])


def test():
    blstm = BiLSTMTagger()
    input_data = "train.conll"
    test_data = "test.conll"
    blstm.train(input_data)
    blstm.accuracy(test_data)
    sentences, tags = blstm.classify(test_data)
