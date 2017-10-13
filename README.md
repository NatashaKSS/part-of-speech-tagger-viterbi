#### Done by: Natasha Koh
#### Matric no.: A0130894B

# CS4248 Assignment 2
Implementation of a Part-of-Speech tagger using the Viterbi algorithm with the Penn Treebank tag set.

### Instructions
Running the Viterbi part-of-speech tagger
```
python build_tagger.py sents.train sents.devt model_file
python run_tagger.py sents.test model_file sents.out

# For 10-fold cross validation
# -- BEWARE this might take some time
python cross_validator.py sents.train
```

### File Structure
```
.
├── /build_tagger.py         # Executes the training phase of the tagger on sents.train
├── /run_tagger.py           # Executes the viterbi tagger on sents.test
├── /HMMProbGenerator.py     # Generates the model and computes the resulting P(w_i | t_i) and P(t_i | t_i-1) probabilities
├── /PennTreebankPOSTags.py  # Store of all POS tags used
├── /POSTagger.py            # Executes the viterbi & backpointer algorithms to generate the best POS tags
├── /POSTagModelTrainer      # Loads the training data and executes HMMProbGenerator to generate the model
├── /Tokenizer.py             # Tokenizes the training set, test set and dataset used in CrossValidator
├── /cross_validator.py       # Computes the 10-fold cross validation accuracy of the trained model
└── README.md
```

Thank you!

\- END OF README -
