"""Defines dataset class."""

import logging
from typing import List, Tuple, Generator, Type

import torch

from pycrf.nn.utils import sort_and_pad
from .vocab import Vocab


logger = logging.getLogger(__name__)


SourceType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
TargetType = Type[torch.Tensor]


class Dataset:
    """Class for abstracting training and testing datasets."""

    def __init__(self) -> None:
        self.source: List[SourceType] = []
        self.target: List[TargetType] = []

    def __getitem__(self, key: int) -> Tuple[SourceType, TargetType]:
        return self.source[key], self.target[key]

    def __iter__(self) -> Generator[Tuple[SourceType, TargetType], None, None]:
        for src, tgt in zip(self.source, self.target):
            yield src, tgt

    def __len__(self) -> int:
        return len(self.source)

    def __bool__(self) -> bool:
        return len(self.source) > 0

    def append(self, src: SourceType, tgt: TargetType) -> None:
        """Append a new training example."""
        self.source.append(src)
        self.target.append(tgt)

    def load_file(self,
                  fname: str,
                  vocab: Vocab,
                  limit: int = None,
                  device: torch.device = None) -> None:
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

        vocab : pycrf.io.Vocab
            The vocab instance to apply to the sentences.

        limit : int, optional
            If set, will only load this many examples.

        device : torch.device, optional
            The device to send the tensors to.

        Returns
        -------
        None

        """
        i = 0
        with open(fname, "r") as datafile:
            src: List[str] = []
            tgt: List[str] = []
            for line in datafile.readlines():
                line_list = line.rstrip().split('\t')
                if len(line_list) == 1:  # end of sentence.
                    # Get target tensor.
                    target_tensor = vocab.labs2tensor(tgt)
                    if device is not None:
                        target_tensor = target_tensor.to(device)
                    self.target.append(target_tensor)

                    # Get source tensors.
                    char_tensors, word_lengths, word_tensors = \
                        vocab.sent2tensor(src)
                    # ``char_tensors`` is a list. Sort and pad to turn it into
                    # a single tensor.
                    sorted_char_tensors, lens, idxs = \
                        sort_and_pad(char_tensors, word_lengths)
                    if device is not None:
                        sorted_char_tensors, lens, idxs, word_tensors = \
                            sorted_char_tensors.to(device), \
                            lens.to(device), \
                            idxs.to(device), \
                            word_tensors.to(device)
                    self.source.append(
                        (sorted_char_tensors,
                         lens,
                         idxs,
                         word_tensors)
                    )

                    src = []
                    tgt = []
                    i += 1
                    if limit is not None and i == limit:
                        break
                else:
                    src.append(line_list[0])
                    tgt.append(line_list[1])
