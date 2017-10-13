"""
PENN TREEBANK POS TAG SET
A store of all POS tags used for this assignment.

Summary of additional tag symbols:
<S>      e.g. start-of-sentence marker
<E>      e.g. end-of-sentence marker
``       e.g. left quote "
''       e.g. right quote "
-LRB-    e.g. (, [, {
-RRB-    e.g. ), ], }
"""

START_MARKER = '<S>'
END_MARKER = '<E>'

POS_TAGS = [
  START_MARKER, END_MARKER,
  'CC', 'CD', 'DT', 'EX', 'FW', 'IN',
  'JJ', 'JJR', 'JJS', 'LS', 'MD',
  'NN', 'NNS', 'NNP', 'NNPS',
  'PDT', 'POS', 'PRP', 'PRP$',
  'RB', 'RBR', 'RBS', 'RP',
  'SYM', 'TO', 'UH',
  'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
  'WDT', 'WP', 'WP$', 'WRB',
  '$', '#', '``', '\'\'', '-LRB-', '-RRB-', ',', '.', ':'
]
