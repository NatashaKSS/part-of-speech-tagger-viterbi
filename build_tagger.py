# Import standard modules
import sys

# Import custom modules
from Tokenizer import Tokenizer;
from PennTreebankPOSTags import POS_TAGS;

#=====================================================#
# TRAINING THE POS TAGGER
#=====================================================#
class POSTagModelTrainer():
  def __init__(self):
    print('__init__')
    self.tokenizer = Tokenizer()

  def train(self):
    list_of_str_postag = self.load_training_data()

  """
  Loads the training data in-memory and splits each token on ' '

  return        List of strings in the format '<word>/<pos tag>'
  """
  def load_training_data(self):
    DATA_TRAIN = open(PATH_TO_DATA_TRAIN).read()
    list_of_str_postag = self.tokenizer.tokenize_document(DATA_TRAIN)
    list_of_word_postag_pairs = self.tokenizer.get_pairs_of_word_tags(list_of_str_postag)

#=====================================================#
# EXECUTION OF PROGRAM
#=====================================================#
PATH_TO_DATA_TRAIN = sys.argv[1]
PATH_TO_DATA_DEVT = sys.argv[2]
PATH_TO_DATA_MODEL = sys.argv[3]

print("Training data:", PATH_TO_DATA_TRAIN + ", Devt Data:", PATH_TO_DATA_DEVT + ", Model file:", PATH_TO_DATA_MODEL)

model = POSTagModelTrainer().train()
