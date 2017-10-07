# Import standard modules
import sys
from math import log

# Import custom modules
from PennTreebankPOSTags import POS_TAGS, START_MARKER, END_MARKER

# Define constants
UNK = '<UNK>' # symbol representing out-of-vocabulary words

class HMMProbGenerator():
  def __init__(self, word_postag_pairs):
    print("HMMProbGenerator instantiated...")
    self.WORD_POSTAG_PAIRS = word_postag_pairs

    #==================================================#
    # Constructing the Vocabulary for words & tags
    #==================================================#
    # Vocabulary in this Dictionary format: { 'the': 41107, 'gracious': 1, ... }
    self.WORD_VOCAB = self.get_word_vocabulary_with_counts(word_postag_pairs)

    # Vocabulary in this Dictionary format: { 'NN': 1123, 'VBN': 2323, ... }
    self.POSTAG_VOCAB = self.get_tag_vocabulary_with_counts(word_postag_pairs)

    #==================================================#
    # Initialize probabilities required for the model
    #==================================================#
    # Matrix representing P(t_i | t_i-1), where rows: t_i-1, cols: t_i
    self.PROB_TAG_GIVEN_TAG = self.initialize_prob_tag_given_tag()

    # Matrix representing P(w_i | t_i),  where rows: t_i, cols: w_i
    self.PROB_WORD_GIVEN_TAG = self.initialize_word_given_tag()

  #=======================================================#
  # GENERATE P(t_i | t_i-1) AND P(w_i | t_i) PROBABILITIES
  #=======================================================#
  """
  Generate emission & transition probabilities from a labelled corpus
  Modifies self.PROB_TAG_GIVEN_TAG and self.PROB_WORD_GIVEN_TAG

  return    Model as a list of 2 elements [P(t_i | t_i-1), P(w_i | t_i)]
  """
  def generate_probs(self):
    self.generate_prob_word_given_tag()
    self.generate_prob_tag_given_tag()
    return [self.PROB_TAG_GIVEN_TAG, self.PROB_WORD_GIVEN_TAG]

  """
  Generates P(t_i | t_i-1) bigram tags' occurrence probability matrix
  Modifies self.PROB_TAG_GIVEN_TAG
  """
  def generate_prob_tag_given_tag(self):
    # Count number of tags at position (i) following another tag at position (i + 1)
    WORD_POSTAG_PAIRS_LENGTH = len(self.WORD_POSTAG_PAIRS)
    for i in range(WORD_POSTAG_PAIRS_LENGTH):
      if i + 1 < WORD_POSTAG_PAIRS_LENGTH: # prevents index out of bounds
        bi_tag_i_minus_1 = self.WORD_POSTAG_PAIRS[i][1]
        bi_tag_i = self.WORD_POSTAG_PAIRS[i + 1][1]
        self.PROB_TAG_GIVEN_TAG[bi_tag_i_minus_1][bi_tag_i] += 1

    # Convert to probability from raw counts
    for tag_i_minus_1 in self.PROB_TAG_GIVEN_TAG:
      for tag_i in self.PROB_TAG_GIVEN_TAG[tag_i_minus_1]:
        if self.POSTAG_VOCAB[tag_i_minus_1] != 0:
          # Raw counts to probability
          self.PROB_TAG_GIVEN_TAG[tag_i_minus_1][tag_i] = \
            float(self.PROB_TAG_GIVEN_TAG[tag_i_minus_1][tag_i]) / float(self.POSTAG_VOCAB[tag_i_minus_1])

        # Convert entry to log probability
        value = self.PROB_TAG_GIVEN_TAG[tag_i_minus_1][tag_i]
        self.PROB_TAG_GIVEN_TAG[tag_i_minus_1][tag_i] = log(value) if value != 0.0 else log(sys.float_info.min)

    return None

  """
  Generates P(w_i | t_i) word and POS tag occurrence probability matrix
  Modifies self.PROB_WORD_GIVEN_TAG

  Also handles out-of-vocabulary words by assigning them with a count of
  1 for every tag
  """
  def generate_prob_word_given_tag(self):
    # Count number of words co-occurring with a given tag & mutate PROB_WORD_GIVEN_TAG matrix
    for word_postag_pair in self.WORD_POSTAG_PAIRS:
      pair_word = word_postag_pair[0]
      pair_postag = word_postag_pair[1]

      self.PROB_WORD_GIVEN_TAG[pair_postag][pair_word] += 1

    # Set count of out-of-vocabulary words to 1, normalized probability
    for postag in self.PROB_WORD_GIVEN_TAG:
      self.PROB_WORD_GIVEN_TAG[postag][UNK] += 1

    # Convert to probability from raw counts
    for postag in self.PROB_WORD_GIVEN_TAG:
      for word in self.PROB_WORD_GIVEN_TAG[postag]:
        if self.POSTAG_VOCAB[postag] != 0: # prevents division by 0 errors
          # Raw counts to probability
          self.PROB_WORD_GIVEN_TAG[postag][word] = \
            float(self.PROB_WORD_GIVEN_TAG[postag][word]) / float(self.POSTAG_VOCAB[postag])
        else:
          self.PROB_WORD_GIVEN_TAG[postag][word] = 0

        # Convert entry to log probability
        value = self.PROB_WORD_GIVEN_TAG[postag][word]
        self.PROB_WORD_GIVEN_TAG[postag][word] = log(value) if value != 0.0 else log(sys.float_info.min)

    return None

  #=====================================================#
  # INITIALIZE ALL PROBABILITY MATRICES NEEDED FOR VITERBI
  #=====================================================#
  """
  Initializes matrix representing bigram tags' occurrence probabilities, i.e.
  P(t_i | t_i-1).
  Rows: t_i-1
  Cols: t_i

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
  Rows: t_i
  Cols: w_i

  return    Dictionary of POS tags at (i)th position with nested Dictionary
            of words at (i)th position
  """
  def initialize_word_given_tag(self):
    prob_word_given_tag = {}
    for postag in POS_TAGS:
      prob_word_given_tag[postag] = {}
      prob_word_given_tag[postag][UNK] = 0
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
