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

  def get_test_data_tokens(self, doc_string):
    return re.findall(r"[\w]+|[.,!?;\(\)\[\](...)('s)('d)(n't)('ll)('S)('D)(N'T)('LL)]+", doc_string)

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

  def get_train_data_tokens(self, doc_string_with_S_E_tags):
    return doc_string_with_S_E_tags.split(' ')

  def get_sentences(self, doc_string):
    return doc_string.split('\n')

  def insert_start_end_sentence_tags(self, sentences):
    START_SENTENCE = NEWLINE + '/' + START_MARKER + ' '
    END_SENTENCE = ' ' + NEWLINE + '/' + END_MARKER
    return [START_SENTENCE + sentence.strip(' ') + END_SENTENCE for sentence in sentences]

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
