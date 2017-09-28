# Import standard modules
import sys
import math

# Import custom modules
from PennTreebankPOSTags import POS_TAGS

class HMMProbGenerator():
  def __init__(self, word_postag_pairs):
    print("HMMProbGenerator instantiated...")

    # Initialized constants
    self.WORD_POSTAG_PAIRS = word_postag_pairs
    self.WORD_VOCAB = self.get_word_vocabulary_with_counts(word_postag_pairs)
    self.POSTAG_VOCAB = self.get_tag_vocabulary_with_counts(word_postag_pairs)
    self.PROB_TAG_GIVEN_TAG = self.initialize_prob_tag_given_tag()
    self.PROB_WORD_GIVEN_TAG = self.initialize_word_given_tag()

  #=====================================================#
  # VITERBI ALGORITHM
  #=====================================================#
  """
  Generate emission & transition probabilities from a labelled corpus in this
  format: [['its', 'PRP$'], ['to', 'TO'] ...]

  return
  """
  def generate_probs(self):
    self.generate_prob_word_given_tag()
    print(self.PROB_WORD_GIVEN_TAG)
    return 'probs'

  #=====================================================#
  # GENERATE P(t_i | t_i-1) AND P(w_i | t_i) PROBABILITIES
  #=====================================================#
  def generate_prob_tag_given_tag(self):

    return None

  def generate_prob_word_given_tag(self):
    # Count number of words co-occurring with a given tag & mutate PROB_WORD_GIVEN_TAG matrix
    for word_postag_pair in self.WORD_POSTAG_PAIRS:
      pair_word = word_postag_pair[0]
      pair_postag = word_postag_pair[1]

      self.PROB_WORD_GIVEN_TAG[pair_postag][pair_word] += 1

    # Convert to log probability from raw counts
    for postag in self.PROB_WORD_GIVEN_TAG:
      for word in self.PROB_WORD_GIVEN_TAG[postag]:
        if self.POSTAG_VOCAB[postag] != 0: # prevents division by 0 errors
          # Probability to raw counts
          self.PROB_WORD_GIVEN_TAG[postag][word] = \
            self.log_base_10(self.PROB_WORD_GIVEN_TAG[postag][word] / self.POSTAG_VOCAB[postag])
        else:
          self.PROB_WORD_GIVEN_TAG[postag][word] = 0

  def count_word_tagged_as_tag(self, word, tag):
    count = 0
    for word_postag_pair in self.WORD_POSTAG_PAIRS:
      pair_word = word_postag_pair[0]
      pair_postag = word_postag_pair[1]

      if pair_word == word and pair_postag == tag:
        count += 1
    return count

  #=====================================================#
  # INITIALIZE ALL PROBABILITY MATRICES NEEDED FOR VITERBI
  #=====================================================#
  """
  Initializes matrix representing bigram tags' occurrence probabilities, i.e.
  P(t_i | t_i-1)

  return    Dictionary of POS tags at (i-1)th position with nested Dictionary
            of POS tags at (i)th position
  """
  def initialize_prob_tag_given_tag(self):
    prob_tag_given_tag = {}
    for tag_i_minus_1 in POS_TAGS:
      prob_tag_given_tag[tag_i_minus_1] = {}
      for tag_i in POS_TAGS:
        prob_tag_given_tag[tag_i_minus_1][tag_i] = 0
    return prob_tag_given_tag

  """
  Initializes matrix representing word and POS tag occurrence probabilities, i.e.
  P(w_i | t_i)

  return    Dictionary of POS tags at (i)th position with nested Dictionary
            of words at (i)th position
  """
  def initialize_word_given_tag(self):
    prob_word_given_tag = {}
    for postag in POS_TAGS:
      prob_word_given_tag[postag] = {}
      for word in self.WORD_VOCAB:
        prob_word_given_tag[postag][word] = 0
    return prob_word_given_tag

  #=====================================================#
  # INITIALIZE ALL WORD & POS TAG VOCABULARIES NEEDED FOR VITERBI
  #=====================================================#
  """
  Returns the corpus' seen words vocabulary

  return    Vocabulary in this Dictionary format: { 'the': 41107, 'gracious': 1, ... }
  """
  def get_word_vocabulary_with_counts(self, word_postag_pairs):
    result = {}

    # Count seen words & add to our Words vocab
    for word_postag in word_postag_pairs:
      word = word_postag[0]
      if word in result:
        result[word] = result[word] + 1
      else:
        result[word] = 1
    return result

  """
  Returns the corpus' seen tags vocabulary

  return    Vocabulary in this Dictionary format: { 'NN': 1123, 'VBN': 2323, ... }
  """
  def get_tag_vocabulary_with_counts(self, word_postag_pairs):
    result = {}

    # Initialize Tags vocab with all possible POS tag types
    for POS_TAG in POS_TAGS:
      result[POS_TAG] = 0

    # Count Tags & add to our Tags vocab
    for word_postag in word_postag_pairs:
      postag = word_postag[1]
      if postag in result: # check if key exists, just in case, although unneeded
        result[postag] = result[postag] + 1
    return result

  def log_base_10(self, num):
    if num == 0:
      return 0
    else:
      return math.log(num, 10)
