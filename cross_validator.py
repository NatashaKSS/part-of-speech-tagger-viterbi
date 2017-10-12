# Import standard modules
import sys
import string
import re
import math
import pickle

# Import custom modules
from PennTreebankPOSTags import POS_TAGS
from PennTreebankPOSTags import START_MARKER
from PennTreebankPOSTags import END_MARKER
from Tokenizer import Tokenizer
from HMMProbGenerator import HMMProbGenerator
from POSTagModelTrainer import POSTagModelTrainer
from POSTagger import POSTagger

class CrossValidator():
  def __init__(self, PATH_TO_DATA_TRAIN):
    print('== [CrossValidator instantiated] ==')

    # Set up tokenizer before everything else
    self.tokenizer = Tokenizer()

    # Set up the Model Trainer
    self.POS_trainer = POSTagModelTrainer(PATH_TO_DATA_TRAIN, True)

    # Initialized constants
    self.DATA_TRAIN = open(PATH_TO_DATA_TRAIN).read()

  #=====================================================#
  # 10-FOLD CROSS VALIDATION
  #=====================================================#
  def validate(self):
    print('Validating model...please wait...')

    sentences = self.tokenizer.get_sentences(self.DATA_TRAIN)
    LEN_SENTENCES = len(sentences)
    ONE_FOLD_SIZE = int(math.floor(0.1 * LEN_SENTENCES))

    acc_scores_so_far = []
    for i in range(10):
      print('Performing validation on fold no.:', i + 1, 'please wait...')
      sentences = self.shift(sentences, ONE_FOLD_SIZE)
      test_sentences = sentences[-ONE_FOLD_SIZE:]
      training_sentences = sentences[0:-ONE_FOLD_SIZE]

      # Tokenizing Training data
      training_dataset = self.tokenizer.flatten_list_of_sentences(training_sentences)
      list_of_str_postag = self.tokenizer.tokenize_document(training_dataset)
      list_of_word_postag_pairs = self.tokenizer.get_pairs_of_word_tags(list_of_str_postag)

      # Training the model
      model = HMMProbGenerator(list_of_word_postag_pairs).generate_probs()

      # Running the POS Tagger
      self.POS_tagger = POSTagger('', '', model, True)

      # Run the model on the test data
      best_postags_and_gold_standard_tags = self.POS_tagger.run_with_provided_sentences(test_sentences)
      best_postags = best_postags_and_gold_standard_tags[0]
      gold_standard_tags = best_postags_and_gold_standard_tags[1]

      # Compute accuracy
      acc_scores_so_far.append(self.compute_accuracy(gold_standard_tags, best_postags))
      print('Done validation on fold no.:', i + 1, '!', 10 - i, 'more to go!')

    print(acc_scores_so_far)
    print("Average Cross Validation Score:", self.get_average(acc_scores_so_far))

  """
  Shifts a list by n spaces to the right and returns a copy of that array

  arr   List to shift
  n     Number of elements to shift by to the right

  return
  """
  def shift(self, arr, n):
    return arr[n:] + arr[:n]

  """
  Computes the accuracy of our predicted postags as compared to a list of
  true positive postags.

  true_postags    List of true positive POS tags in the format
                  [['<S>', 'FW', 'FW', 'FW', 'NNP', 'NNP', '<E>'], ...]
  test_postags    List of predicted POS tags in the format
                  [['<S>', 'FW', 'FW', 'FW', 'NNP', 'NNP', '<E>'], ...], that
                  is, a list of sentences where each sentence is a list of
                  every word's tag

  return          Percentage accuracy of our model on the testset in a single
                  10-fold cross validation runthrough
  """
  def compute_accuracy(self, true_postags, test_postags):
    N = self.compute_accuracy_N(test_postags)
    correct = 0

    for i in range(len(test_postags)):
      for j in range(len(test_postags[i])):
        if test_postags[i][j] == true_postags[i][j]:
          correct += 1

    return float(correct) / float(N)

  """
  Helper function for compute_accuracy to compute the total number of tags in
  our entire test set

  test_postags    List of predicted POS tags in the format
                  [['<S>', 'FW', 'FW', 'FW', 'NNP', 'NNP', '<E>'], ...], that
                  is, a list of sentences where each sentence is a list of every
                  word's tag

  return          Total number of tags in the entire test set
  """
  def compute_accuracy_N(self, test_postags):
    N = 0
    for sentence in test_postags:
      for postag in sentence:
        N = N + 1
    return N

  """
  Computes the average of a list of numbers

  scores    List of numbers. In this context, it's the list of accuracy scores
            after cross-validation

  return    Average of a list of numbers
  """
  def get_average(self, scores):
    total = 0
    for score in scores:
      total += score
    return float(total) / len(scores)

#=====================================================#
# EXECUTION OF PROGRAM
#=====================================================#
PATH_TO_DATA_TRAIN = sys.argv[1]

print("Path to training data:", PATH_TO_DATA_TRAIN)

CrossValidator(PATH_TO_DATA_TRAIN).validate()
