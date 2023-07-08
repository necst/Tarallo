import numpy as np
import os
from config.config_paths import general_paths
# Load all the needed dicts for the conversions


word2id = np.load(os.path.join(general_paths['dir'], 'DictionaryForRawData/word2id.npz'), allow_pickle=True)
word2id = word2id['word2id'][()]

word2behavior = np.load(os.path.join(general_paths['dir'], 'DictionaryForRawData/word2behavior.npz'), allow_pickle=True)
word2behavior = word2behavior['word2behavior'][()]

behavior2id = np.load(os.path.join(general_paths['dir'], 'DictionaryForRawData/behavior2id.npz'), allow_pickle=True)
behavior2id = behavior2id['behavior2id'][()]

# Reverse map
id2word = dict(map(reversed, word2id.items()))