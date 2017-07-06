import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # TODO implement model selection based on BIC scores
        #Set up blank variable; BIC will be set up at inf due to its maximization to be negative
        best_score, best_model  = float("inf"), None
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                # set up model varying by n_components
                model=self.base_model(n_components)
                # get size of row
                number_of_sample=self.X.shape[0]
                # get size of column
                number_of_features=self.X.shape[1]
                # calculate number of total free parameters
                number_of_parameters=n_components * n_components +2*number_of_features*n_components -1
                # BIC Formula
                bic=-2*model.score(self.X,self.lengths)+np.log(number_of_sample)*number_of_parameters
                # Store if BIC is less than best score
                if bic < best_score:
                    best_score, best_model = bic, model
            except Exception as e:
                return self.base_model(self.n_constant)
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        #Set up blank variable; DIC will be set up at -inf due to its maximization to be positive
        best_score, best_model  = float("-inf"), None
        #Create list of other words
        other_words = list(self.words)
        other_words.remove(self.this_word)
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model=self.base_model(n_components)
                score = model.score(self.X, self.lengths)
                score_other_word=0
                #Accumulate score for other words 
                for i in other_words:
                    X, lengths = self.hwords[i]
                    score_other_word = score_other_word+model.score(X, lengths)
                #DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
                dic=score-(score_other_word/(len(self.words)-1))
                # Store if BIC is greather than best score
                if dic > best_score:
                    best_score, best_model = dic, model
            except Exception as e:
                return self.base_model(self.n_constant)
        return best_model
    
class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection using CV
        #Set up blank variable; CV will be set up at -inf due to its maximization to be positive
        best_score, best_model  = float("-inf"), None
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                #If sequence is 2 or less, then it can't be fold
                if (len(self.lengths)<=2):
                    model = self.base_model(n_components)
                    score = model.score(self.X, self.lengths)
                else:
                    sum_score=0
                    parts_count=0
                    #Split the data into parts
                    split_method=KFold(random_state=self.random_state,n_splits=min(len(self.lengths),3))
                    #Gather sum_score for each parts.
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        #Train Model using X_train
                        model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        #Score model using x_test
                        sum_score = sum_score+model.score(X_test,lengths_test)
                        parts_count = parts_count+1
                    score = sum_score / parts_count
                if score > best_score:    
                    best_score, best_model = score, model
            except Exception as e:
                return self.base_model(self.n_constant)
        return best_model