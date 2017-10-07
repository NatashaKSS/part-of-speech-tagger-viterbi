# Import standard modules
import sys
import string
import re
import math

# Import custom modules
from PennTreebankPOSTags import POS_TAGS
from PennTreebankPOSTags import START_MARKER
from PennTreebankPOSTags import END_MARKER
from Tokenizer import Tokenizer
from HMMProbGenerator import HMMProbGenerator
from POSTagModelTrainer import POSTagModelTrainer

# Define constants

class CrossValidator():
  def __init__(self, PATH_TO_DATA_TRAIN):
    print("Cross Validator instantiated...")

    # Set up tokenizer before everything else
    self.tokenizer = Tokenizer()

    # Set up the POS tagger
    self.POS_tagger = POSTagModelTrainer(PATH_TO_DATA_TRAIN, validate=True)

    # Initialized constants
    self.DATA_TRAIN = open(PATH_TO_DATA_TRAIN).read()

  #=====================================================#
  # PREP DATASETS FOR 10-FOLD CROSS VALIDATION
  #=====================================================#
  def validate(self):
    print('-- Cross Validation of the model --')

    sentences = self.tokenizer.get_sentences(self.DATA_TRAIN)
    LEN_SENTENCES = len(sentences)
    ONE_FOLD_SIZE = int(math.floor(0.1 * LEN_SENTENCES))

    for i in range(10):
      sentences = self.shift(sentences, ONE_FOLD_SIZE)
      test_sentences = sentences[-ONE_FOLD_SIZE:]
      training_sentences = sentences[0:-ONE_FOLD_SIZE]

      # Tokenizing Training data
      training_dataset = self.tokenizer.flatten_list_of_sentences(training_sentences)
      list_of_str_postag = self.tokenizer.tokenize_document(training_dataset)
      list_of_word_postag_pairs = self.tokenizer.get_pairs_of_word_tags(list_of_str_postag)

      # Training the model
      model = HMMProbGenerator(list_of_word_postag_pairs).generate_probs()

  """
  Shifts a list by n spaces to the right and returns a copy of that array

  arr   List to shift
  n     Number of elements to shift by to the right

  return
  """
  def shift(self, arr, n):
    return arr[n:] + arr[:n]

#=====================================================#
# EXECUTION OF PROGRAM
#=====================================================#
PATH_TO_DATA_TRAIN = sys.argv[1]

print("Path to training data:", PATH_TO_DATA_TRAIN)

CrossValidator(PATH_TO_DATA_TRAIN).validate()
