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

    # TODO: We process each sentence as a separate test doc to run the tagger on
    # (might not be accurate especially for shorter articles)
    sen_as_tokens_list = self.generate_tokens_for_test_doc_sentences(sentences)
    for sen_tokens in [sen_as_tokens_list[0]]:
      print(self.tag(sen_tokens))

  #=====================================================#
  # VITERBI ALGORITHM
  #=====================================================#
  def tag(self, tokens):
    LEN_TOKENS = len(tokens)
    LEN_POSTAG = len(POS_TAGS)

    print(tokens)

    # Initialize memo
    memo = {}
    for i in range(LEN_TOKENS):
      memo[i] = [None] * LEN_POSTAG
      for j in range(LEN_POSTAG):
        memo[i][j] = 0

    for i in range(LEN_POSTAG):
      # Probability at '<S>' is 1 since it occurs for all documents
      memo[0][i] = 1

    self.print_viterbi_state(memo, 2, tokens)

    # Compute most probable path
    for i in range(LEN_TOKENS):
      for j in range(LEN_POSTAG):
        if i != 0 and j != 0 and j != LEN_POSTAG - 1: # exclude the first '<S>' word/tag and last state '<E>' word/tag
          if tokens[i] in self.PROB_WORD_GIVEN_TAG[POS_TAGS[j]]:
            memo[i][j] = memo[i][j - 1] * \
                         self.PROB_TAG_GIVEN_TAG[POS_TAGS[j - 1]][POS_TAGS[j]] * \
                         self.PROB_WORD_GIVEN_TAG[POS_TAGS[j]][tokens[i]]

    self.print_viterbi_state(memo, 2, tokens)

    return None

  def print_viterbi_state(self, memo, print_till_token_pos, tokens):
    for i in range(len(tokens)):
      if i < print_till_token_pos:
        print_result = tokens[i] + ' '
        for j in range(len(POS_TAGS)):
          print_result = print_result + str(memo[i][j]) + ' '
        print(print_result)

  #=====================================================#
  # GENERATE TOKENS
  #=====================================================#
  def generate_tokens_for_test_doc_sentences(self, sentences):
    sen_as_tokens_list = [self.tokenizer.tokenize_test_document(sentence) for sentence in sentences]
    sen_as_tokens_list = [sen_tokens for sen_tokens in sen_as_tokens_list if sen_tokens != []] # remove any empty lists due to empty sentences
    return sen_as_tokens_list

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
