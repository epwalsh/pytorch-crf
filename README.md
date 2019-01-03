# pytorch-crf

:exclamation: NOTE: I no longer maintain this repository. I recommend using [AllenNLP](https://github.com/allenai/allennlp) instead.

A [PyTorch](https://pytorch.org/) implementation of a Bi-LSTM CRF with character-level features.

**pytorch-crf** is a flexible framework that makes it easy to reproduce several state-of-the-art sequence labelling deep neural networks that have proven to excel at the tasks of named entity recognition (NER) and part-of-speech (POS) tagging, among others. Some examples of the models you can reproduce with **pytorch-crf** are:
- the LSTM-CRF with LSTM-generated character-level features from [Lample et. al., 2016](https://www.aclweb.org/anthology/N16-1030).
- the CNN-LSTM-CRF with CNN-generated character-level features from [Ma & Hovy, 2016](https://arxiv.org/pdf/1603.01354.pdf).

## Requirements

First and foremost, you will need Python >= 3.6. The remaining requirements are listed in `requirements.txt` and can be installed with `pip install -r requirements.txt`.

## Quick start

In order to train a model, you will need a dataset and pretrained word embeddings.
The dataset should be formatted like this:

```
EU	B-ORG
rejects	O
German	B-MISC
call	O
to	O
boycott	O
British	B-MISC
lamb	O
...
```

Each token is followed by a tab character and then the corresponding label.
The end of a sentence is indicated by a blank line.

The pretrained word embeddings should be stored in text file like this:

```
the -0.038194 -0.24487 0.72812 -0.39961 ...
, -0.10767 0.11053 0.59812 -0.54361 ...
. -0.33979 0.20941 0.46348 -0.64792 ...
of -0.1529 -0.24279 0.89837 0.16996 ...
```

Each line contains a term in the vocabulary, followed by a space, and then the embedding coordinates (separated by spaces).

A model is trained using the **pytorch-crf** command-line interface, which is invoked with `python -m pycrf.train`. You can see all of the
available options with `python -m pycrf.train --help`.

## Sources

- [Chiu, J. P. C., and Nichols, E. 2016. Named entity recognition with bidirectional lstm-cnns. TACL](https://arxiv.org/pdf/1511.08308.pdf)
- [Huang, Z., Xu, W., and Yu, K. 2015. Bidirectional lstm-crf models for sequence tagging. arXiv preprint arXiv:1508.01991.](https://arxiv.org/pdf/1508.01991.pdf)
- [Lample, G., Ballesteros, M., Kawakami, K., Subramanian, S., and Dyer, C. 2016. Neural architectures for named entity recognition. In NAACL-HLT.](https://www.aclweb.org/anthology/N16-1030)
- [Ma, X., and Hovy, E. 2016. End-to-end sequence labeling via bi-directional lstm-cnns-crf. In ACL.](https://arxiv.org/pdf/1603.01354.pdf)
