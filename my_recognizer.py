import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for i in range(test_set.num_items):
        best_prob, best_word = float("-inf"), None
        prob=dict()
        X, lengths = test_set.get_item_Xlengths(i)
        for word, model in models.items():
            try:
                prob[word] = model.score(X, lengths)
            except Exception as e:
                prob[word] = float("-inf")
            if prob[word] > best_prob:
                best_prob, best_word = prob[word], word
        probabilities.append(prob)
        guesses.append(best_word)
    return probabilities, guesses