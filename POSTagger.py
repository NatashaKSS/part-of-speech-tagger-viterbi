# Import standard modules
import sys
import math
import pickle

# Import custom modules
from Tokenizer import Tokenizer
from PennTreebankPOSTags import POS_TAGS, START_MARKER, END_MARKER

# Define constants
UNK = '<UNK>'

#=====================================================#
# RUNNING THE POS TAGGER
#=====================================================#
class POSTagger():
  def __init__(self, PATH_TO_DATA_TEST, PATH_TO_DATA_MODEL, model=None, VALIDATE_MODE=False):
    MODEL = None
    if VALIDATE_MODE:
      print("== [POSTagger instantiated] CROSS VALIDATION MODE ==")
      MODEL = model
    else:
      print("== [POSTagger instantiated] ==")
      self.PATH_TO_DATA_TEST = PATH_TO_DATA_TEST
      self.PATH_TO_DATA_MODEL = PATH_TO_DATA_MODEL
      MODEL = self.load_model()

    # Matrix representing P(t_i | t_i-1), where rows: t_i-1, cols: t_i
    self.PROB_TAG_GIVEN_TAG = MODEL[0]

    # Matrix representing P(w_i | t_i),  where rows: t_i, cols: w_i
    self.PROB_WORD_GIVEN_TAG = MODEL[1]

    # List of seen words
    self.VOCAB_WORDS = self.PROB_WORD_GIVEN_TAG['NN'].keys()

    self.tokenizer = Tokenizer()

  def run(self):
    sentences = self.load_document_as_sentences()
    print(self.get_best_postags(sentences))

  def run_with_provided_sentences(self, sentences):
    return self.get_best_postags_for_cross_validation(sentences)

  def get_best_postags(self, sentences):
    print("-- RUNNING THE PART OF SPEECH TAGGER --")
    sen_as_tokens_list = self.generate_tokens_for_test_doc_sentences(sentences, self.VOCAB_WORDS)

    best_postags_list = []
    for i in range(len(sen_as_tokens_list)):
      best_postags = self.tag(sen_as_tokens_list[i])
      best_postags_list.append(best_postags)
    return best_postags_list

  # sentences = ['<S>/<S> The/DT ...', '<S>/<S> The/DT']
  def get_best_postags_for_cross_validation(self, sentences):
    print("-- RUNNING THE PART OF SPEECH TAGGER FOR CROSS VALIDATION --")
    sentences = self.tokenizer.insert_start_end_sentence_tags(sentences)
    test_sentences_and_tags = self.tokenizer.extract_tags_from_test_dataset(sentences, self.VOCAB_WORDS)
    test_sentences = test_sentences_and_tags[0]
    test_tags = test_sentences_and_tags[1]

    sen_as_tokens_list = self.generate_tokens_for_test_doc_sentences(test_sentences, self.VOCAB_WORDS)

    best_postags_list = []
    for i in range(len(sen_as_tokens_list)):
      best_postags = self.tag(sen_as_tokens_list[i])
      best_postags_list.append(best_postags)
    return (best_postags_list, test_tags)

  #=====================================================#
  # VITERBI ALGORITHM
  #=====================================================#
  def tag(self, tokens):
    LEN_TOKENS = len(tokens)
    LEN_POSTAG = len(POS_TAGS)

    # print(len(tokens), 'tokens like this: ', tokens)

    # Initialize memo & backpointers for the best tags

    # Visualize memo as a HMM network laid out
    # Visualize best_postags as a HMM network laid out, denoting each viterbi
    # node's best chosen backpointer to some previous viterbi node
    memo = []
    best_postags = []
    for i in range(LEN_TOKENS):
      memo.append([])
      best_postags.append([])
      for j in range(LEN_POSTAG):
        memo[i].append(sys.float_info.min)
        best_postags[i].append(-1)

        if i == 0:
          # Initialize probability at '<S>' to 1 since it occurs for all documents
          memo[i][j] = 0  # (log scale equivalent of probability = 1 is 0)

    # Compute most probable path & store in memo and best_postags arrays
    for i in range(1, LEN_TOKENS):
      for j in range(LEN_POSTAG):
        curr_max = -sys.float_info.max
        back_ptr = -1

        for k in range(LEN_POSTAG):
          transition_prob = memo[i - 1][k] + \
                            self.PROB_TAG_GIVEN_TAG[POS_TAGS[k]][POS_TAGS[j]] + \
                            self.PROB_WORD_GIVEN_TAG[POS_TAGS[j]][tokens[i]]

          if transition_prob > curr_max:
            curr_max = transition_prob
            back_ptr = k if i > 1 else 0

        memo[i][j] = curr_max
        best_postags[i][j] = back_ptr

    best_postag_at_end_of_sentence = self.find_arg_of_max(memo[LEN_TOKENS - 1])
    found_best_postag_value = best_postag_at_end_of_sentence[0]
    found_best_postag_index = best_postag_at_end_of_sentence[1]

    return self.get_best_viterbi_path(best_postags, found_best_postag_index)

  def get_best_viterbi_path(self, back_ptrs, best_end_of_sentence_back_ptr):
    # to traverse a sentence from last POS tag to 1st POS tag
    # Note: ignore the 1st tag since its back ptr is undefined (i.e. there's no
    #       reasonable back ptr for the 1st tag of a sentence)
    back_ptrs = list(reversed(back_ptrs[1:]))
    LEN_BACK_PTRS = len(back_ptrs)

    # last POS TAG is always an END_MARKER
    # we can say this because we always add a START_MARKER & END_MARKER between
    # sentences during the tokenization phase of the test set
    best_pos_tag_sequence = [END_MARKER]

    for i in range(LEN_BACK_PTRS): # traversing sentence backwards
      best_pos_tag_sequence.append(POS_TAGS[back_ptrs[i][best_end_of_sentence_back_ptr]])
      best_end_of_sentence_back_ptr = back_ptrs[i][best_end_of_sentence_back_ptr]

    # remember to reverse sequence of best POS tags since we traversed sentence backwards
    return list(reversed(best_pos_tag_sequence))

  #=====================================================#
  # GENERATE TOKENS
  #=====================================================#
  # sentences in the format: ['<S> The cow...', '<S> The man...', ...]
  def generate_tokens_for_test_doc_sentences(self, sentences, word_vocab):
    sen_as_tokens_list = [self.tokenizer.tokenize_test_document(sentence, word_vocab) for sentence in sentences]
    sen_as_tokens_list = [sen_tokens for sen_tokens in sen_as_tokens_list if sen_tokens != []] # remove any empty lists due to empty sentences
    return sen_as_tokens_list

  #=====================================================#
  # LOAD FILES & INITIALIZE MODEL
  #=====================================================#
  def load_document_as_sentences(self):
    DATA_TEST = open(self.PATH_TO_DATA_TEST).read()
    return self.tokenizer.generate_sentences_from_test_document(DATA_TEST)

  def load_model(self):
    return pickle.load(open(self.PATH_TO_DATA_MODEL, 'rb'))

  #=====================================================#
  # HELPER METHODS
  #=====================================================#
  def find_arg_of_max(self, lst):
    curr_max = -sys.float_info.max
    curr_max_index = -1

    for i in range(len(lst)):
      if lst[i] > curr_max:
        curr_max = lst[i]
        curr_max_index = i

    return (curr_max, curr_max_index)
