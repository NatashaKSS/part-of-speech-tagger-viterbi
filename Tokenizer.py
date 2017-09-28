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

  return        List of tokens split according to the RegEX rules
  """
  def tokenize_document(self, doc_string):
    print(doc_string)
    return re.findall(r"[\w]+|[.,!?;\(\)\[\](...)('s)('d)(n't)]+", doc_string);
