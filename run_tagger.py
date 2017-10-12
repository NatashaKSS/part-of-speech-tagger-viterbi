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

print("sents.test:", PATH_TO_DATA_TEST + ", model_file:", PATH_TO_DATA_MODEL + ", labelled test data sents.out:", PATH_TO_DATA_TEST_LABELLED)

# Get the best POS tags for the test set
output = POSTagger(PATH_TO_DATA_TEST, PATH_TO_DATA_MODEL).run()

# Print to an output file. In this assignment, it is called 'sents.out'
with open(PATH_TO_DATA_TEST_LABELLED, 'w') as sents_out_file:
  sents_out_file.write(output)
