from typing import List, Tuple
import numpy as np
import torch
import torchtext
from torch import nn
from torch.nn import functional


def vocabularize(dataset: List[Tuple[str, str]]):
    """Take input/target pairs, produce a vocabulary and one hot everything"""
    # Tokenize
    tokenizer = torchtext.data.utils.get_tokenizer('spacy')
    tokenized = [(tokenizer(i), tokenizer(t)) for i, t in dataset]
    # Build vocab
    def generator():
        for i, t in tokenized:
            yield i
            yield t
    vocab = torchtext.vocab.build_vocab_from_iterator(generator)
    # One hotify
    s = len(vocab)
    all_indices = [[vocab(i), vocab(t)] for i, t in tokenized]
    return nn.functional.one_hot(torch.tensor(all_indices), s), vocab