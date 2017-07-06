
import os
os.getcwd()
os.chdir('C:/Users/warezt/Source/Repos/AIND-Recognizer')
os.getcwd()

import numpy as np
import pandas as pd
from asl_data import AslDb
asl = AslDb() # initializes the database
asl.df.head() # displays the first five rows of the asl database, indexed by video and frame
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']

df_means = asl.df.groupby('speaker').mean()

asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])

from asl_utils import test_std_tryit
# TODO Create a dataframe named `df_std` with standard deviations grouped by speaker
df_std = asl.df.groupby('speaker').std()


# TODO add features for normalized by speaker values of left, right, x, y
# Name these 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
# using Z-score scaling (X-Xmean)/Xstd
asl.df['right-x-mean']= asl.df['speaker'].map(df_means['right-x'])
asl.df['right-y-mean']= asl.df['speaker'].map(df_means['right-y'])
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
asl.df['left-y-mean']= asl.df['speaker'].map(df_means['left-y'])
features_mean=['right-x-mean','right-y-mean','left-x-mean','left-y-mean']
asl.df['right-x-std']= asl.df['speaker'].map(df_std['right-x'])
asl.df['right-y-std']= asl.df['speaker'].map(df_std['right-y'])
asl.df['left-x-std']= asl.df['speaker'].map(df_std['left-x'])
asl.df['left-y-std']= asl.df['speaker'].map(df_std['left-y'])
features_std=['right-x-std','right-y-std','left-x-std','left-y-std']
asl.df['norm-rx']=(asl.df['right-x']-asl.df['right-x-mean'])/asl.df['right-x-std']
asl.df['norm-ry']=(asl.df['right-y']-asl.df['right-y-mean'])/asl.df['right-y-std']
asl.df['norm-lx']=(asl.df['left-x']-asl.df['left-x-mean'])/asl.df['left-x-std']
asl.df['norm-ly']=(asl.df['left-y']-asl.df['left-y-mean'])/asl.df['left-y-std']
features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']


# TODO add features for polar coordinate values where the nose is the origin
# Name these 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
# Note that 'polar-rr' and 'polar-rtheta' refer to the radius and angle
import numpy as np
asl.df['polar-rr']=np.sqrt((asl.df['grnd-rx']**2)+(asl.df['grnd-ry']**2))
asl.df['polar-lr']=np.sqrt((asl.df['grnd-lx']**2)+(asl.df['grnd-ly']**2))
asl.df['polar-rtheta']=np.arctan2(asl.df['grnd-rx'],asl.df['grnd-ry'])
asl.df['polar-ltheta']=np.arctan2(asl.df['grnd-lx'],asl.df['grnd-ly'])
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

# TODO add features for left, right, x, y differences by one time step, i.e. the "delta" values discussed in the lecture
# Name these 'delta-rx', 'delta-ry', 'delta-lx', and 'delta-ly'
asl.df['delta-rx']=asl.df['grnd-rx'].diff().fillna(0)
asl.df['delta-ry']=asl.df['grnd-ry'].diff().fillna(0)
asl.df['delta-lx']=asl.df['grnd-lx'].diff().fillna(0)
asl.df['delta-ly']=asl.df['grnd-ly'].diff().fillna(0)
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

# TODO add features of your own design, which may be a combination of the above or something else
# Name these whatever you would like
asl.df['delta-norm-rx']=((asl.df['right-x']-asl.df['right-x-mean'])/asl.df['right-x-std']).diff().fillna(0)
asl.df['delta-norm-ry']=((asl.df['right-y']-asl.df['right-y-mean'])/asl.df['right-y-std']).diff().fillna(0)
asl.df['delta-norm-lx']=((asl.df['left-x']-asl.df['left-x-mean'])/asl.df['left-x-std']).diff().fillna(0)
asl.df['delta-norm-ly']=((asl.df['left-y']-asl.df['left-y-mean'])/asl.df['left-y-std']).diff().fillna(0)

# TODO define a list named 'features_custom' for building the training set
features_custom=['delta-norm-rx','delta-norm-ry','delta-norm-lx','delta-norm-ly']


from my_recognizer import recognize
from asl_utils import show_errors
from my_model_selectors import SelectorCV
from my_model_selectors import SelectorBIC
from my_model_selectors import SelectorDIC

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word, 
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict
# TODO Choose a feature set and model selector
features = features_delta # change as needed
model_selector = SelectorCV # change as needed

# TODO Recognize the test set and display the result with the show_errors method
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)