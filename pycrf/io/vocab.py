"""Defines a class for holding a vocabulary set."""

import string
from typing import List, Tuple, Type

import torch
import torchtext

from pycrf.nn.utils import sort_and_pad


SourceType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
TargetType = Type[torch.Tensor]


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

    glove : :obj:``torchtext.vocab.GloVe``
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
                 word_vec_dim: int = 100,
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
                    sent: List[str],
                    device: torch.device = None) -> SourceType:
        # pylint: disable=not-callable
        """
        Transform a sentence into a tensor.

        Parameters
        ----------
        sent : List[str]
            The sentence to transform.

        device : torch.device, optional
            The device to send the tensors to.

        Returns
        -------
        SourceType
            A tuple where the first item is the sorted words in their character
            representation. The second item is the lengths of those words as
            defined by the number of characters. The third item is the sorted
            ids of the words so that the original order can be retained, and the
            fourth is an unsorted tensor of word embeddings.

        """
        # Encode words and characters.
        word_lengths_list: List[int] = []           # length of each word.
        word_vec_emb_list: List[torch.Tensor] = []  # words represented by their vector embedding.
        word_tensors_list: List[torch.Tensor] = []  # words represented by their characters.
        for tok in sent:
            word_lengths_list.append(len(tok))
            word_vec_emb_list.append(
                self.glove[self.glove.stoi.get(tok.lower(), -1)])
            tmp = torch.tensor([
                self.chars_stoi.get(char, self.chars_stoi[self.unk_char])
                for char in tok
            ])
            word_tensors_list.append(tmp)

        # Convert into tensors.
        word_lengths = torch.tensor(word_lengths_list)
        word_vec_emb = torch.cat(word_vec_emb_list, dim=0)
        # Words are sorted and padded by word length.
        word_tensors, lens, idxs = \
            sort_and_pad(word_tensors_list, word_lengths)

        # Send to device (like a cuda device) if device was given.
        if device is not None:
            word_tensors, lens, idxs, word_vec_emb = \
                word_tensors.to(device), \
                lens.to(device), \
                idxs.to(device), \
                word_vec_emb.to(device)

        return word_tensors, lens, idxs, word_vec_emb

    def labs2tensor(self,
                    labs: List[str],
                    device: torch.device = None) -> TargetType:
        # pylint: disable=not-callable
        """
        Transform a list of labels to a tensor.

        Parameters
        ----------
        labs : list of str
            The list of labels to transform.

        device : torch.device, optional
            The device to send the tensors to.

        Returns
        -------
        torch.Tensor
            The tensor of integers corresponding to the list of labels.

        """
        target = torch.tensor([self.labels_stoi.get(lab, 0) for lab in labs])
        if device is not None:
            target = target.to(device)
        return target
