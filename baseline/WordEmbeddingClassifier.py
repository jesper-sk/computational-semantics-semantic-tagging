import fasttext
import fasttext.util
import os
import data


class WordEmbeddingClassifier:
    """Base class for the SVM and DT taggers since they use the same word
    embedding
    """
    def __init__(self, lang: str) -> None:
        self.lang: str = lang
        self.word_embedding: fasttext.FastText = self.__get_word_embedding()

    def __get_word_embedding(self) -> fasttext.FastText:
        """Get a pretrained word embedding. Downloads it first if necessary."""

        # Save current wd
        current_wd = os.getcwd()

        # Set name of model file
        model_fname = f'./models/cc.{self.lang}.300.bin'

        # Download model to ./models/ subfolder
        os.makedirs('./models/', exist_ok=True)
        os.chdir('./models')
        fasttext.util.download_model(self.lang)

        # Remove compressed model to save some space
        if os.path.exists(f'{model_fname}.gz'):
            os.remove(f'{model_fname}.gz')

        # Change back to original wd
        os.chdir(current_wd)

        # Return model
        return fasttext.load_model(model_fname)

    def prepare_data(self, input_data):
        """Transform a data set of words into vectors"""

        in_raw, out = data.dt_data(input_data)
        # Prepare the data by transforming words to vectors. This is slow
        # because FastText doesn't have a vectorized way to do this (i.e. we
        # would ideally do self.__word_embedding[in_raw])
        in_vectors = [self.word_embedding[word] for word in in_raw]
        return in_raw, in_vectors, out
