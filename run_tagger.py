# Import standard modules
import sys
import math
import pickle

# Import custom modules
from Tokenizer import Tokenizer
from PennTreebankPOSTags import POS_TAGS, START_MARKER, END_MARKER

#=====================================================#
# RUNNING THE POS TAGGER
#=====================================================#
class POSTagger():
  def __init__(self):
    # Initialize constants
    MODEL = self.load_model()
    self.PROB_TAG_GIVEN_TAG = MODEL[0]
    self.PROB_WORD_GIVEN_TAG = MODEL[1]

    self.tokenizer = Tokenizer()

  def run(self):
    print("-- RUNNING THE PART OF SPEECH TAGGER --")
    sentences = self.load_document_as_sentences()
    sen_as_tokens_list = [self.tokenizer.tokenize_test_document(sentence) for sentence in sentences]
    sen_as_tokens_list = [sen_tokens for sen_tokens in sen_as_tokens_list if sen_tokens != []] # remove any empty lists due to empty sentences

    for sen_tokens in sen_as_tokens_list:
      viterbi_memo = self.initialize_viterbi_memo_matrix(sen_tokens)
      print(viterbi_memo['NN'])

  #=====================================================#
  # VITERBI ALGORITHM
  #=====================================================#
  def initialize_viterbi_memo_matrix(self, tokens):
    memo = {}
    for postag in POS_TAGS:
      for token in tokens:
        if postag not in memo:
          memo[postag] = {}
        memo[postag][token] = float('-inf')
    return memo

  def get_POS_tags(self, tokens):
    print(tokens)

  #=====================================================#
  # LOAD FILES & INITIALIZE MODEL
  #=====================================================#
  def load_document_as_sentences(self):
    DATA_TEST = open(PATH_TO_DATA_TEST).read()
    return self.tokenizer.generate_sentences_from_test_document(DATA_TEST)

  def load_model(self):
    return pickle.load(open(PATH_TO_DATA_MODEL, 'rb'))

#=====================================================#
# EXECUTION OF PROGRAM
#=====================================================#
PATH_TO_DATA_TEST = sys.argv[1]
PATH_TO_DATA_MODEL = sys.argv[2]
PATH_TO_DATA_TEST_LABELLED = sys.argv[3]

print("Test data:", PATH_TO_DATA_TEST + ", Model:", PATH_TO_DATA_MODEL + ", labelled test data:", PATH_TO_DATA_TEST_LABELLED)

tagger = POSTagger().run()
