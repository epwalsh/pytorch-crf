"""Defines a class for holding a vocabulary set."""

import logging
import string
from typing import List, Tuple

import torch
import torchtext


logger = logging.getLogger(__name__)


class Vocab:
    """
    Class for abstracting a set of vocabulary.

    Basically just wraps a dict object for terms and dict object for
    characters.

    Parameters
    ----------
    labels : list of str
        List of valid token labels that we expect to see in the datasets.

    default_label : str
        The default target label.

    unk_term : str
        The code to use for unknown terms.

    unk_char : str
        The code to use for unknown chars.

    pad_char : str
        The code to use for padding chars.

    word_vec_dim : int
        The dimension of the vector word embedding. As of now, can only be 50,
        100, 200, or 300.

    cache : str
        Optional path to GloVe cache. If not given and the cache cannot be
        found automatically, the GloVe embeddings will be downloaded.

    Attributes
    ----------
    labels : list of str
        The sequence target labels (what we try to predict).

    default_label : str
        The default target label.

    labels_stoi : dict
        Keys are labels, values are label ids.

    labels_itos : dict
        Keys are label ids, values are labels.

    unk_term : str
        The code to use for unknown terms.

    unk_char : str
        The code to use for unknown chars.

    pad_char : str
        The code to use for padding chars.

    glove : :obj:`torchtext.vocab.GloVe`
        GloVe word embeddings.

    chars_stoi : dict
        Keys are chars, values are unique ids (int).

    chars_itos : dict
        Keys are ids (int), values are chars.

    """

    def __init__(self,
                 labels: List[str],
                 default_label: str = "O",
                 unk_term: str = "UNK",
                 pad_char: str = "PAD",
                 unk_char: str = "UNK",
                 word_vec_dim: int = 300,
                 cache: str = None) -> None:
        self.default_label = default_label
        self.labels_stoi = {default_label: 0}
        self.labels_itos = {0: default_label}

        self.unk_term = unk_term
        self.unk_char = unk_char
        self.pad_char = pad_char

        self.word_vec_dim = word_vec_dim

        self.chars_stoi = {pad_char: 0, unk_char: 1}
        self.chars_itos = {0: pad_char, 1: unk_char}

        self.glove = torchtext.vocab.GloVe(
            name="6B", dim=self.word_vec_dim, cache=cache)

        for lab in labels:
            ind = self.labels_stoi.setdefault(lab, len(self.labels_stoi))
            self.labels_itos[ind] = lab

        for char in string.ascii_letters + string.digits + string.punctuation:
            ind = self.chars_stoi.setdefault(
                char, len(self.chars_stoi))
            self.chars_itos.setdefault(ind, char)
            ind = self.chars_stoi.setdefault(
                char.upper(), len(self.chars_stoi))
            self.chars_itos.setdefault(ind, char.upper())

    @property
    def n_words(self) -> int:
        """Get the number of words."""
        return len(self.glove.stoi)

    @property
    def n_chars(self) -> int:
        """Get the number of characters."""
        return len(self.chars_stoi)

    @property
    def n_labels(self) -> int:
        """Get the number of target labels."""
        return len(self.labels_itos)

    def sent2tensor(self,
                    sent: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform a sentence into a tensor.

        Parameters
        ----------
        sent : list of str
            The sentence to transform.

        Returns
        -------
        tuple (list of :obj:`torch.Tensor`, :obj:`torch.Tensor`)
            The first item is a list of length `len(sent)` of tensors, each of
            which has size `[len(sent[i]) x self.n_chars]`.
            The second item has is a tensor of size
            `[len(sent) x self.word_vec_dim]`.

        """
        char_tensors = []
        word_tensors = []
        for tok in sent:
            word_tensors.append(
                self.glove[self.glove.stoi.get(tok.lower(), -1)])
            tmp_list = []
            for char in tok:
                tmp = torch.zeros(self.n_chars)
                tmp[self.chars_stoi.get(char, 0)] = 1
                tmp_list.append(tmp.view(1, -1))
            char_tensors.append(torch.cat(tmp_list))
        return char_tensors, torch.cat(word_tensors, dim=0)

    def labs2tensor(self, labs: List[str]) -> torch.Tensor:
        """
        Transform a list of labels to a tensor.

        Parameters
        ----------
        labs : list of str
            The list of labels to transform.

        Returns
        -------
        :obj:`torch.Tensor`
            The tensor of integers corresponding to the list of labels.

        """
        # pylint: disable=not-callable
        return torch.tensor([self.labels_stoi.get(lab, 0) for lab in labs])
