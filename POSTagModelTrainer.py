# Import standard modules
import sys
import pickle

# Import custom modules
from Tokenizer import Tokenizer
from HMMProbGenerator import HMMProbGenerator

#=====================================================#
# TRAINING THE POS TAGGER
#=====================================================#
class POSTagModelTrainer():
  def __init__(self, PATH_TO_DATA_TRAIN, VALIDATE_MODE=False):
    if VALIDATE_MODE:
      print("== [POSTagModelTrainer instantiated] CROSS VALIDATION MODE ==")
    else:
      print("== [POSTagModelTrainer instantiated] ==")
      # Set up tokenizer before everything else
      self.tokenizer = Tokenizer()

      # Initialized constants
      self.DATA_TRAIN = open(PATH_TO_DATA_TRAIN).read()
      self.LIST_OF_WORD_POSTAG_PAIRS = self.load_training_data()

  def train(self):
    list_of_labelled_words = self.LIST_OF_WORD_POSTAG_PAIRS # [['its', 'PRP$'], ['to', 'TO'] ...]
    model = HMMProbGenerator(list_of_labelled_words).generate_probs()
    return model

  """
  Loads the training data in-memory and tokenizes it.

  return        List of strings in the format
                ['<word1>/<pos_tag1>', '<word2>/<pos_tag2>', ...]
  """
  def load_training_data(self):
    list_of_str_postag = self.tokenizer.tokenize_document(self.DATA_TRAIN)
    list_of_word_postag_pairs = self.tokenizer.get_pairs_of_word_tags(list_of_str_postag)
    return list_of_word_postag_pairs
