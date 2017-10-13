# Import standard modules
import sys
import math
import pickle

from PennTreebankPOSTags import POS_TAGS, START_MARKER, END_MARKER

#=====================================================#
# EXECUTION OF PROGRAM
#=====================================================#
errors = pickle.load(open('cross-validation-results-investigate', 'rb'))

for postag in POS_TAGS:
  if postag in errors.keys():
    print('==', postag, '==')
    print(errors[postag])
