# Import standard modules
import sys
import pickle

# Import custom modules
from PennTreebankPOSTags import POS_TAGS
from Tokenizer import Tokenizer
from HMMProbGenerator import HMMProbGenerator

#=====================================================#
# TRAINING THE POS TAGGER
#=====================================================#
class POSTagModelTrainer():
  def __init__(self):
    # Set up tokenizer before everything else
    self.tokenizer = Tokenizer()

    # Initialized constants
    self.DATA_TRAIN = open(PATH_TO_DATA_TRAIN).read()
    self.LIST_OF_WORD_POSTAG_PAIRS = self.load_training_data()

  def train(self):
    list_of_labelled_words = self.LIST_OF_WORD_POSTAG_PAIRS # [['its', 'PRP$'], ['to', 'TO'] ...]
    model = HMMProbGenerator(list_of_labelled_words).generate_probs()
    pickle.dump(model, open('model_file', 'wb'))
    print("--- FINISHED TRAINING...MODEL SAVED IN model_file ---")

  """
  Loads the training data in-memory and splits each token on ' '

  return        List of strings in the format ['<word1>/<pos_tag1>', '<word2>/<pos_tag2>', ...]
  """
  def load_training_data(self):
    list_of_str_postag = self.tokenizer.tokenize_document(self.DATA_TRAIN)
    list_of_word_postag_pairs = self.tokenizer.get_pairs_of_word_tags(list_of_str_postag)
    return list_of_word_postag_pairs


  #=====================================================#
  # DEBUGGING HELPER FUNCTIONS
  #=====================================================#
  def findTokenPair(self, word1):
    for index, postag_pairs in enumerate(self.LIST_OF_WORD_POSTAG_PAIRS):
      if postag_pairs[0] == word1:
        return [index, postag_pairs[0], postag_pairs[1]]
    return None

#=====================================================#
# EXECUTION OF PROGRAM
#=====================================================#
PATH_TO_DATA_TRAIN = sys.argv[1]
PATH_TO_DATA_DEVT = sys.argv[2]
PATH_TO_DATA_MODEL = sys.argv[3]

print("Training data:", PATH_TO_DATA_TRAIN + ", Devt Data:", PATH_TO_DATA_DEVT + ", Model file:", PATH_TO_DATA_MODEL)

model = POSTagModelTrainer().train()
