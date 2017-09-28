# Import standard modules
import string
import re

# Import custom modules
from PennTreebankPOSTags import START_MARKER
from PennTreebankPOSTags import END_MARKER

NEWLINE = 'NEWLINE' # represents the char \n

class Tokenizer():
  def __init__(self):
    print("Tokenizer instantiated...")

  #=====================================================#
  # TOKENIZER FOR UNSEEN SENTENCES
  #=====================================================#
  def generate_sentences_from_test_document(self, doc_string):
    sentences = self.get_sentences(doc_string)
    return self.insert_start_end_sentence_tokens(sentences)

  """
  Tokenizes a string for the Test set.

  doc_string    String of the document to tokenize

  return    List of tokens split according to the RegEX rules
  """
  def tokenize_test_document(self, doc_string):
    doc_tokens = self.get_test_data_tokens(doc_string)
    return self.remove_empty_sentence_at_end(doc_tokens)

  # Removes the 'S' and 'E' at the back of [...'<S>', '', '<E>'].
  # This occurs when we have an extra sentence at the last line in a file
  def remove_empty_sentence_at_end(self, tokens):
    LENGTH_TOKENS = len(tokens)
    if LENGTH_TOKENS >= 3:
      if tokens[LENGTH_TOKENS - 3] == START_MARKER and tokens[LENGTH_TOKENS - 1] == END_MARKER:
        # in the format of [...'<S>', '', '<E>']
        return tokens[0 : -3]

    # 1-word tokens will never have empty sentences at the end
    return tokens

  # Sandwich START & END of sentence markers between each sentence in a list of sentences
  # Any leading and trailing space between sentences will be removed too
  # Unlike the tokenizer for labelled corpus, we DON'T add the labelled START & END MARKER
  def insert_start_end_sentence_tokens(self, sentences):
    START_SENTENCE = START_MARKER + ' '
    END_SENTENCE = ' ' + END_MARKER
    return [START_SENTENCE + sentence.strip(' ') + END_SENTENCE for sentence in sentences]

  # Terms in test data are separated by spaces, so this splits them
  def get_test_data_tokens(self, doc_string_with_S_E):
    return doc_string_with_S_E.split(' ')

  # == UNUSED ==
  # Tokenizes some raw corpus based on these RegEX rules.
  # Since the test set is in a well-defined format, for this assignment, we need
  # not use these rules. Splitting on ' ' is fine.
  def get_test_data_tokens_UNUSED(self, doc_string):
    return re.findall(r"[\w]+|[.,!?;\(\)\[\](...)('s)('d)(n't)('ll)('S)('D)(N'T)('LL)]+", doc_string)

  #=====================================================#
  # TOKENIZER FOR LABELLED CORPUS
  #=====================================================#
  """
  Tokenizes a string close to the Penn TreeBank tokenizer format.
  Extracts whole words, punctuations and apostrophes.

  doc_string    String of the document to tokenize

  return    List of tokens split according to the RegEX rules
  """
  def tokenize_document(self, doc_string):
    sentences = self.get_sentences(doc_string)
    sentences_S_E_tags = self.insert_start_end_sentence_tags(sentences)
    doc_str_with_S_E_tags = self.flatten_list_of_sentences(sentences_S_E_tags)
    return self.get_train_data_tokens(doc_str_with_S_E_tags)

  # Terms in training data are separated by spaces, so this splits them
  def get_train_data_tokens(self, doc_string_with_S_E_tags):
    return doc_string_with_S_E_tags.split(' ')

  # Sentences in training data are separated by '\n', so this splits them
  def get_sentences(self, doc_string):
    return doc_string.split('\n')

  # Sandwich START & END of sentence markers between each sentence in a list of sentences
  # Any leading and trailing space between sentences will be removed too
  def insert_start_end_sentence_tags(self, sentences):
    START_SENTENCE = START_MARKER + '/' + START_MARKER + ' '
    END_SENTENCE = ' ' + END_MARKER + '/' + END_MARKER
    return [START_SENTENCE + sentence.strip(' ') + END_SENTENCE for sentence in sentences]

  # Stitch each sentence into 1 big String
  def flatten_list_of_sentences(self, sentences):
    return ' '.join(sentences)

  """
  Tokenizes a word & pos_tag. Ignores empty strings if they exist.

  list_of_str_postag    ['perhaps/RB', 'forced/VBN' ...]

  return    List of token pairs [['perhaps', 'RB'], ['forced', 'VBN'] ...]
  """
  def get_pairs_of_word_tags(self, list_of_str_postag):
    return [self.__convert_to_word_postag_pair(str_postag) for str_postag in list_of_str_postag if len(str_postag) > 0]

  # Converts a 'perhaps/RB' string to ['perhaps', 'RB']
  def __convert_to_word_postag_pair(self, str_postag):
    str_postag_pair = str_postag.rsplit('/', 1)
    return [str_postag_pair[0], str_postag_pair[1]]
