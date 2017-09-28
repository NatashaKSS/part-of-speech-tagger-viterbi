# Import standard modules
import sys

from Tokenizer import Tokenizer;

#=====================================================#
# TRAINING THE POS TAGGER
#=====================================================#
class POSTagModelTrainer():
  def __init__(self):
    print('__init__')
    self.tokenizer = Tokenizer()

  def load_training_data(self):
    DATA_TRAIN = open(PATH_TO_DATA_TRAIN).read().split()
    print(DATA_TRAIN)


#=====================================================#
# EXECUTION OF PROGRAM
#=====================================================#
PATH_TO_DATA_TRAIN = sys.argv[1]
PATH_TO_DATA_DEVT = sys.argv[2]
PATH_TO_DATA_MODEL = sys.argv[3]

print("Training data:", PATH_TO_DATA_TRAIN + ", Devt Data:", PATH_TO_DATA_DEVT + ", Model file:", PATH_TO_DATA_MODEL)

model = POSTagModelTrainer()
model.load_training_data()
