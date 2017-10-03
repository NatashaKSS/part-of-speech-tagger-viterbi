# CS4248 Assignment 2
Implementation of a POS tagger using the viterbi algorithm with the Penn Treebank tag set.

### Instructions

Run
```
python build_tagger.py sents.train sents.devt model_file
python run_tagger.py sents.test model_file sents.out
```

#### Handling unknown words
**Strategy:** Add a new symbol `<UNK>` to represent unseen words in test data
For `P(w_i | t_i)`, set it to
`self.PROB_WORD_GIVEN_TAG[pair_postag][pair_word] = 1*k / self.POSTAG_VOCAB[postag]`
where `k` is a constant that can be tuned in the future

**Extension:** Could add another symbol for words appearing less than some threshold number of times, like `<FEW>`.
