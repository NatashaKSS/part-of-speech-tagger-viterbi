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
    memo = []
    for i in range(LEN_TOKENS):
      memo.append([])
      for j in range(LEN_POSTAG):
        memo[i].append(0)

    for i in range(LEN_POSTAG):
      # Probability at '<S>' is 1 since it occurs for all documents
      memo[0][i] = 1

    # Smoothing parameter
    smooth_k = self.add_1_smoothing(tokens)

    # Compute most probable path
    best_indices = []
    for i in range(LEN_POSTAG):
      best_indices.append(0)

    self.print_viterbi_state(memo, 4, tokens)

    for i in range(1, LEN_TOKENS):
      for j in range(LEN_POSTAG):

        curr_max = 0
        for k in range(LEN_POSTAG):
          # print(tokens[i], POS_TAGS[j], 'v:', memo[i - 1][j], ', P(t_i-1, t_i):', self.PROB_TAG_GIVEN_TAG[POS_TAGS[j]][POS_TAGS[j]], ', P(w_i, t_i):', self.PROB_WORD_GIVEN_TAG[POS_TAGS[j]][tokens[i]])
          transition_prob = memo[i - 1][k] * self.PROB_TAG_GIVEN_TAG[POS_TAGS[k]][POS_TAGS[j]]

          if transition_prob > curr_max:
            curr_max = transition_prob

        if tokens[i] in self.PROB_WORD_GIVEN_TAG[POS_TAGS[j]]:
          # For seen words
          memo[i][j] = curr_max * self.PROB_WORD_GIVEN_TAG[POS_TAGS[j]][tokens[i]]
        else:
          # For unseen words
          # # TODO: Implement way to handle unseen tokens
          memo[i][j] = curr_max * 0

    self.print_viterbi_state(memo, len(tokens), tokens)
    # print(best_indices)

    return None

  # res = 'Labor: '
  # headers = ' '
  # for postag in POS_TAGS:
  #   headers = headers + str(postag) + ' '
  #   res = res + str(self.PROB_WORD_GIVEN_TAG[postag]['Labor']) + ' '
  # print(headers)
  # print(res)

  def print_viterbi_state(self, memo, print_till_token_pos, tokens):
    for i in range(len(tokens)):
      if i < print_till_token_pos:
        print_result = tokens[i] + ' '
        for j in range(len(POS_TAGS)):
          print_result = print_result + str(memo[i][j]) + ' '
        print(print_result)

  #=====================================================#
  # TODO: SMOOTHING
  # GOAL:
  #   * Ensure probability of unseen words are not 0, or else, HMM will not work
  #
  # STRATEGY:
  #   * Treat unseen words as though we've only seen them once
  #     To penalize them even further, we could even make that probability only half as much, etc.
  #=====================================================#
  def add_1_smoothing(self, tokens):
    unseen_count_pair = self.count_unseen_words(tokens)
    vocab_size = self.count_seen_words() + unseen_count_pair[0]
    seen_unseen_tokens = [unseen_count_pair[1], self.PROB_WORD_GIVEN_TAG[START_MARKER].keys()]
    trained_tokens = [item for sublist in seen_unseen_tokens for item in sublist]
    return 1 / len(trained_tokens)

  def count_seen_words(self):
    return len(self.PROB_WORD_GIVEN_TAG[START_MARKER].keys())

  def count_unseen_words(self, tokens):
    count = 0
    unseens = []
    for token in tokens:
      if token not in self.PROB_WORD_GIVEN_TAG[START_MARKER]:
        count += 1
        unseens.append(token)
    return [count, unseens]

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
