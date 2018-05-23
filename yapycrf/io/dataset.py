"""Defines dataset class."""

import logging


logger = logging.getLogger(__name__)


class Dataset:
    """Class for abstracting training and testing datasets."""

    def __init__(self):
        self.source = []
        self.target = []

    def __iter__(self):
        for src, tgt in zip(self.source, self.target):
            yield src, tgt

    def __len__(self):
        return len(self.source)

    def load_file(self, fname, vocab):
        """
        Load sentences from a file.

        Parameters
        ----------
        fname : str
            The path to the file to load. Files are assumed to look like this:

            ::

                Hi     O
                there  O

                how    O
                are    O
                you    O
                ?      O

            Each sentence is followed by an empty line, and each line
            corresponding to a token in the sentence begins with the token,
            then a tab character, then the corresponding label.

        vocab : :obj:`yapycrf.io.Vocab`.
            The vocab instance to apply to the sentences.

        Returns
        -------
        None

        """
        with open(fname, "w") as datafile:
            src = []
            tgt = []
            for line in datafile.readlines():
                line = line.rstrip().split('\t')
                if len(line) == 1:  # end of sentence.
                    char_tensors, word_tensors = vocab.sent2tensor(src)
                    target_tensor = vocab.labs2tensor(tgt)
                    self.source.append((char_tensors, word_tensors))
                    self.target.append(target_tensor)
                    src = []
                    tgt = []
                else:
                    src.append(line[0])
                    tgt.append(line[1])
