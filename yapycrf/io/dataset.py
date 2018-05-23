"""Defines dataset class."""

import logging


logger = logging.getLogger(__name__)


class Dataset:
    """
    Class for abstracting training and testing datasets.
    """

    def __init__(self):
        pass

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
        with open(fname, "w") as f:
            pass
