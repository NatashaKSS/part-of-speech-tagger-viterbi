# Import standard modules
import sys
import pickle

# Import custom modules
from POSTagModelTrainer import POSTagModelTrainer

#===========================================================================#
# BUILD_TAGGER
# EXECUTES THE TRAINING PHASE OF THE VITERBI TAGGER ON sents.train & PREPS
# THE MODEL_FILE.
#
# Writes the resulting P(w_i | t_i) and P(t_i | t_i-1) probabilities to a
# Python pickle file to be used for run_tagger.py during testing.
#===========================================================================#
PATH_TO_DATA_TRAIN = sys.argv[1]
PATH_TO_DATA_DEVT = sys.argv[2]
PATH_TO_DATA_MODEL = sys.argv[3]

print("Training data:", PATH_TO_DATA_TRAIN + "Devt Data:", PATH_TO_DATA_DEVT, "Model file:", PATH_TO_DATA_MODEL)

model = POSTagModelTrainer(PATH_TO_DATA_TRAIN).train()

pickle.dump(model, open(PATH_TO_DATA_MODEL, 'wb'))
print("=== FINISHED TRAINING...MODEL SAVED IN " + PATH_TO_DATA_MODEL + " ===")
