# Import standard modules
import string
import re

class Tokenizer():
  def __init__(self):
    print("Tokenizer instantiated...")

  """
  Tokenizes a string close to the Penn TreeBank tokenizer format.
  Extracts whole words, punctuations and apostrophes.

  doc_string    String of the document to tokenize

  return    List of tokens split according to the RegEX rules
  """
  def tokenize_document(self, doc_string):
    return re.findall(r"[\w]+|[.,!?;\(\)\[\](...)('s)('d)(n't)]+", doc_string);

  """
  Tokenizes a word & pos_tag

  list_of_str_postag    ['perhaps/RB', 'forced/VBN' ...]

  return    List of token pairs [['perhaps', 'RB'], ['forced', 'VBN'] ...]
  """
  def get_pairs_of_word_tags(self, list_of_str_postag):
    return list(map(self.__convert_to_word_postag_pair, list_of_str_postag))

  # Converts a 'perhaps/RB' string to ['perhaps', 'RB']
  def __convert_to_word_postag_pair(self, str_postag):
    str_postag_pair = str_postag.split('/')
    return [str_postag_pair[0], str_postag_pair[1]]
