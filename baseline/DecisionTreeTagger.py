from typing import List, Optional, Tuple
import data
from dataclasses import dataclass


@dataclass
class DecisionTreeTaggerOptions:
    """Options for the decision tree tagger.
    """

    # Whether to perform n-fold cross-validation, and if so, for what value of n
    n_fold_cv: int = None


class DecisionTreeTagger:
    """A semantic tagger using decision trees
    """

    def __init__(self, path, options: DecisionTreeTaggerOptions) -> None:
        self.__data = data.read_file(path)

    def split_data(self) -> Tuple[List, List, List]:
        """Creates training, validation and test datasets
        """

    def train(self) -> None:
        """Train a decision tree tagger on some data set
        """

    def classify(self, new_data) -> List:
        """Classify new data after training
        """

if __name__ == "__main__":
    #TODO: some testing code here
    pass
