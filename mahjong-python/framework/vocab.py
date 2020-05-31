# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn

UNK_LABEL = '<unk>'
PAD_LABEL = '<pad>'
SOS_LABEL = '<sos>'
EOS_LABEL = '<eos>'


def collect_ngrams(words, n=2):
    """Return all n-grams of words."""
    ngrams = set()
    for i in range(len(words) - n):
        ngrams.add(' '.join(words[i:i + n]))
    return ngrams


class Vocab:
    def __init__(self, name='anonymous',
                 embedding_dim=None, frozen=False,
                 add_padding=True, add_sos_and_eos=False):
        self.name = name.upper()
        self._words = []
        self._word2index = {}
        self._word_vectors = []
        self.frozen = frozen
        self.embedding_dim = embedding_dim

        self._offset = 0
        if add_padding:
            self._add(PAD_LABEL)
            self._offset += 1
        if add_sos_and_eos:
            self._add(SOS_LABEL)
            self._add(EOS_LABEL)
            self._offset += 2

    def __contains__(self, key):
        return self.get_index(key) is not None

    def __iter__(self):
        return iter(self._words)

    def __len__(self):
        return len(self._words)

    def __str__(self):
        size = len(self)
        shape = (size + 1, self.embedding_dim)
        fmt = '<Vocab {} shape={} size={} {}>'
        return fmt.format(self.name, shape, size, self._words[:self._offset])

    def _add(self, word):
        assert word not in self._word2index
        index = len(self._words)
        self._word2index[word] = index
        self._words.append(word)
        return index

    def load(self, vocab_path):
        word_vectors = []
        for line in open(vocab_path):
            word, *vector = line.strip().split()
            self._add(word)
            if vector:
                vector = np.array(vector, dtype=np.float32)
                word_vectors.append(vector)
                assert len(word_vectors) == len(self._words) - self._offset
        if word_vectors:
            self._word_vectors = word_vectors
            self.embedding_dim = word_vectors[0].size
        self._word_vectors = word_vectors
        self.frozen = True  # freeze vocab after loading
        return self

    def state_dict(self):
        return {
            'name': self.name,
            '_offset': self._offset,
            '_words': self._words,
            'embedding_dim': self.embedding_dim,
            'frozen': self.frozen
        }

    def load_state_dict(self, state):
        self.name = state['name']
        self._offset = state['_offset']
        self._words = state['_words']
        word2index = {}
        for index, word in enumerate(self._words):
            word2index[word] = index
        self._word2index = word2index
        self.embedding_dim = state['embedding_dim']
        self.frozen = state['frozen']

    def create_embedding(self):
        # last row is embedding for UNK_LABEL
        num_embeddings = len(self._words) + 1
        padding_idx = None
        if self._words[0] == PAD_LABEL:
            padding_idx = 0
        embedding = nn.Embedding(num_embeddings, self.embedding_dim, padding_idx=padding_idx)

        if self._word_vectors:
            # except UNK_LABEL
            embedding.weight.data[self._offset:-1] = torch.tensor(self._word_vectors)

        return embedding

    def get_index(self, word):
        return self._word2index.get(word)

    def get_word(self, index):
        if index == len(self._words):
            return UNK_LABEL
        return self._words[index]

    def get_words(self, indices, truncate_to_eos=False):
        eos = self.get_index(EOS_LABEL)
        if truncate_to_eos:
            assert eos is not None
            k = 0
            while k < len(indices) and indices[k] != eos:
                k += 1
            indices = indices[:k]
        return [self.get_word(i) for i in indices]

    def get_vector(self, word):
        index = self.get_index(word)
        if index is not None and index >= self._offset:
            assert self._word_vectors is not None
            return self._word_vectors[index - self._offset]
        return None

    def to_indices(self, words, frozen=False):
        sequence = []
        frozen = frozen or self.frozen
        for word in words:
            index = self.get_index(word)
            if index is None:
                if frozen:  # use UNK_LABEL
                    index = len(self._words)
                else:  # add to vocab
                    index = self._add(word)
            sequence.append(index)
        return sequence

    def to_char_matrix(self, words, max_chars_per_word=-1):
        sequence = []
        for word in words:
            current_seq = self.to_indices(word)
            if max_chars_per_word != -1:
                current_seq = current_seq[:max_chars_per_word]
            sequence.append(current_seq)
        return sequence

    def add(self, word, vector=None):
        index = self.get_index(word)
        if index is None:
            index = self._word2index[word] = len(self._words)
            self._words.append(word)
            if vector is not None:
                self._word_vectors.append(vector)
        return index

    def freeze(self):
        self.frozen = True

    def clear_vectors(self):
        self._word_vectors = None

    def dump(self, filename):
        assert self._word_vectors is not None
        assert len(self._word_vectors) == len(self._words) - self._offset
        with open(filename, 'w') as out:
            for word, vector in zip(self._words[self._offset:], self._word_vectors):
                out.write('{}\t{}\n'.format(word,
                                            ' '.join(str(v) for v in vector)))


class VocabSet(dict):
    def freeze(self):
        for vocab in self.values():
            vocab.freeze()

    def state_dict(self):
        return {name: vocab.state_dict() for name, vocab in self.items()}

    def load_state_dict(self, states):
        self.clear()
        for name, state in states.items():
            vocab = Vocab(name)
            vocab.load_state_dict(state)
            self[name] = vocab
        return self

    def create_embeddings(self):
        return {name: vocab.create_embedding()
                for name, vocab in self.items() if vocab.embedding_dim is not None}

    def new(self, name, **kwargs):
        vocab = Vocab(name, **kwargs)
        self[name] = vocab
        return vocab

    def new_or_default(self, name, **kwargs):
        vocab = self.get(name)
        if vocab is None:
            return self.new(name, **kwargs)
        return vocab
