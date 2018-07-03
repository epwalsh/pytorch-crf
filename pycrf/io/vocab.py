"""Defines a class for holding a vocabulary set."""

import string
from typing import List, Tuple, Type, Dict

import torch

from pycrf.nn.utils import sort_and_pad


SourceType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
TargetType = Type[torch.Tensor]


class Vocab:
    """
    Class for abstracting a set of vocabulary.

    Basically just wraps a dict object for terms and dict object for
    characters.

    Parameters
    ----------
    words_stoi : dict
        Keys are lowercase words, values and ints.

    words_itos : dict
        Keys are ints, values are lowercase words.

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

    chars_stoi : dict
        Keys are chars, values are unique ids (int).

    chars_itos : dict
        Keys are ids (int), values are chars.

    words_stoi : dict
        Keys are lowercase words, values and ints.

    words_itos : dict
        Keys are ints, values are lowercase words.

    sent_context_stoi : dict
        Dict lookup of index of sentence-level context.

    """

    def __init__(self,
                 words_stoi: Dict[str, int],
                 words_itos: Dict[int, str],
                 labels: List[str] = None,
                 default_label: str = "O",
                 default_context: str = "default",
                 unk_term: str = "UNK",
                 pad_char: str = "PAD",
                 unk_char: str = "UNK") -> None:
        self.default_label = default_label
        self.labels_stoi = {default_label: 0}
        self.labels_itos = {0: default_label}
        self.unk_term = unk_term
        self.unk_char = unk_char
        self.pad_char = pad_char
        self.chars_stoi = {pad_char: 0, unk_char: 1}
        self.chars_itos = {0: pad_char, 1: unk_char}
        self.words_stoi = words_stoi
        self.words_itos = words_itos
        self.sent_context_stoi: Dict[str, int] = {default_context: 0}

        for lab in labels or []:
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
        return len(self.words_stoi)

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
                    device: torch.device = None,
                    sent_context: str = None,
                    test: bool = False) -> SourceType:
        # pylint: disable=not-callable
        """
        Transform a sentence into a tensor.

        Parameters
        ----------
        sent : List[str]
            The sentence to transform.

        device : torch.device, optional
            The device to send the tensors to.

        sent_context : str, optional
            The sentence-level context.

        test : bool
            Whether or this is for a new test sentence.

        Returns
        -------
        SourceType
            A tuple where the first item is the sorted words in their character
            representation. The second item is the lengths of those words as
            defined by the number of characters. The third item is the sorted
            indices of the words so that the original order can be retained,
            and the fourth is an unsorted tensor of word ids. The fourth item
            is a scalar value representing the sentence-level context. If no
            context is given, the fourth item will be None.

        """
        # Encode sentence-level context.
        if sent_context is not None:
            if test:
                context_i = self.sent_context_stoi.get(sent_context, 0)
            else:
                context_i = self.sent_context_stoi.setdefault(
                    sent_context, len(self.sent_context_stoi))
                self.sent_context_stoi[sent_context] = context_i
            context = torch.tensor(context_i)
        else:
            context = None

        # Encode words and characters.
        word_lengths_list: List[int] = []           # length of each word.
        word_idx_list: List[torch.Tensor] = []      # words represented by their vector embedding.
        word_tensors_list: List[torch.Tensor] = []  # words represented by their characters.
        for tok in sent:
            word_lengths_list.append(len(tok))
            tmp = torch.tensor([
                self.chars_stoi.get(char, self.chars_stoi[self.unk_char])
                for char in tok
            ])
            word_tensors_list.append(tmp)
            word_idx_list.append(
                self.words_stoi.get(tok.lower(), self.words_stoi[self.unk_term.lower()]))

        # Convert into tensors.
        word_lengths = torch.tensor(word_lengths_list)
        word_idxs = torch.tensor(word_idx_list)
        # Words are sorted and padded by word length.
        word_tensors, lens, idxs = \
            sort_and_pad(word_tensors_list, word_lengths)

        # Send to device (like a cuda device) if device was given.
        if device is not None:
            word_tensors, lens, idxs, word_idxs = \
                word_tensors.to(device), \
                lens.to(device), \
                idxs.to(device), \
                word_idxs.to(device)
            if context is not None:
                context = context.to(device)

        return word_tensors, lens, idxs, word_idxs, context

    def labs2tensor(self,
                    labs: List[str],
                    device: torch.device = None,
                    test: bool = False) -> TargetType:
        # pylint: disable=not-callable
        """
        Transform a list of labels to a tensor.

        Parameters
        ----------
        labs : list of str
            The list of labels to transform.

        device : torch.device, optional
            The device to send the tensors to.

        test : bool
            Whether or this is for a new test sentence.

        Returns
        -------
        torch.Tensor
            The tensor of integers corresponding to the list of labels.

        """
        lab_ids = []
        for lab in labs:
            if test:
                i = self.labels_stoi.get(lab, 0)
            else:
                i = self.labels_stoi.setdefault(lab, len(self.labels_stoi))
                self.labels_itos[i] = lab
            lab_ids.append(i)
        target = torch.tensor(lab_ids)
        if device is not None:
            target = target.to(device)
        return target
