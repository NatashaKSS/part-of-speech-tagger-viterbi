# Import standard modules
import sys
import math
import pickle

# Import custom modules
from Tokenizer import Tokenizer
from PennTreebankPOSTags import POS_TAGS, START_MARKER, END_MARKER
from POSTagger import POSTagger

#=====================================================#
# EXECUTION OF PROGRAM
#=====================================================#
PATH_TO_DATA_TEST = sys.argv[1]
PATH_TO_DATA_MODEL = sys.argv[2]
PATH_TO_DATA_TEST_LABELLED = sys.argv[3]

print("Test data:", PATH_TO_DATA_TEST + ", Model:", PATH_TO_DATA_MODEL + ", labelled test data:", PATH_TO_DATA_TEST_LABELLED)

tagger = POSTagger(PATH_TO_DATA_TEST, PATH_TO_DATA_MODEL).run()
