"""Handles loading of pretrained word vectors."""

from typing import Tuple, Dict, List

import torch


def load_pretrained(path) -> Tuple[torch.Tensor, Dict[int, str], Dict[str, int]]:
    # pylint: disable=not-callable
    """
    Load pretrained word embeddings.

    Parameters
    ----------
    path : str
        The path to the text file.

    Returns
    -------
    Tuple[torch.Tensor, Dict[int, str], Dict[str, int]]
        The word embeddings, the term dict of index-to-str, and the term dict
        of str-to-index.

    """
    vector_list: List[torch.Tensor] = []
    terms_itos: Dict[int, str] = {}
    terms_stoi: Dict[str, int] = {}
    with open(path, "r") as vector_cache:
        for embedding in vector_cache:
            items = embedding.rstrip().split(' ')
            term = items[0]
            index = len(terms_stoi)
            terms_stoi[term] = index
            terms_itos[index] = term
            vector = torch.tensor([float(x) for x in items[1:]])
            vector_list.append(vector.unsqueeze(0))
    return torch.cat(vector_list, dim=0), terms_itos, terms_stoi
