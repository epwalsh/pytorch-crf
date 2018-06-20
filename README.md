# pytorch-crf

A [PyTorch](https://pytorch.org/) implementation of a Bi-LSTM CRF with character-level features.

**pytorch-crf** is a flexible framework that makes it easy to reproduce several
state-of-the-art sequence labelling deep neural networks that have proven to excel
at the tasks of named entity recognition (NER) and part-of-speech (POS) tagging,
among others. Some examples of the models you can reproduce with **pytorch-crf** are:
- the LSTM-CRF with LSTM-generated character-level features from [Lample et. al., 2016](https://www.aclweb.org/anthology/N16-1030).
- the CNN-LSTM-CRF with CNN-generated character-level features from [Ma & Hovy, 2016](https://arxiv.org/pdf/1603.01354.pdf).

> Results on the CoNLL-2003 dataset for NER are consisted with the results stated in the papers.

## Requirements

First and foremost, you will need Python 3.6.
The remaining requirements are listed in `requirements.txt` and can be installed with
`pip install -r requirements.txt`.

## Quick start

In order to train a model, the least you will need is training dataset in text file formated like this:

```
-DOCSTART-	O

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

A model is trained using the **pytorch-crf** command-line interface, which is invoked
with `python -m pycrf.train`. Here is an example of training a model on a small
test dataset:

```
python -m pycrf.train \
    --train ./test/data/sample_dataset.txt
```

## Sources

- [Chiu, J. P. C., and Nichols, E. 2016. Named entity recognition with bidirectional lstm-cnns. TACL](https://arxiv.org/pdf/1511.08308.pdf)
- [Huang, Z., Xu, W., and Yu, K. 2015. Bidirectional lstm-crf models for sequence tagging. arXiv preprint arXiv:1508.01991.](https://arxiv.org/pdf/1508.01991.pdf)
- [Lample, G., Ballesteros, M., Kawakami, K., Subramanian, S., and Dyer, C. 2016. Neural architectures for named entity recognition. In NAACL-HLT.](https://www.aclweb.org/anthology/N16-1030)
- [Ma, X., and Hovy, E. 2016. End-to-end sequence labeling via bi-directional lstm-cnns-crf. In ACL.](https://arxiv.org/pdf/1603.01354.pdf)
