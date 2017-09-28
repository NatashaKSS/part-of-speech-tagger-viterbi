# Import standard modules
import sys

from Tokenizer import Tokenizer;

#=====================================================#
# RUNNING THE POS TAGGER
#=====================================================#
class POSTagger():
  def __init__(self):
    print('__init__')
    self.tokenizer = Tokenizer()

  def load_document(self):
    DATA_TEST = open(PATH_TO_DATA_TEST).read()
    print(self.tokenizer.tokenize_document(DATA_TEST))

#=====================================================#
# EXECUTION OF PROGRAM
#=====================================================#
PATH_TO_DATA_TEST = sys.argv[1]
PATH_TO_DATA_MODEL = sys.argv[2]
PATH_TO_DATA_TEST_LABELLED = sys.argv[3]

print("Test data:", PATH_TO_DATA_TEST + ", Model:", PATH_TO_DATA_MODEL + ", labelled test data:", PATH_TO_DATA_TEST_LABELLED)

tagger = POSTagger()
tagger.load_document()
