import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
from functools import partial
import spacy


class Vocabulary:
    def __init__(self, specials=None):
        if specials is None:
            specials = ['<unk>', '<pad>', '<sos>', '<eos>']
        self.itos = list(specials)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def build_from_counter(self, counter, min_freq=2):
        for word, freq in sorted(counter.items()):
            if freq >= min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)

    def lookup_token(self, idx):
        return self.itos[idx]

    def lookup_indices(self, tokens):
        unk_idx = self.stoi['<unk>']
        return [self.stoi.get(t, unk_idx) for t in tokens]

    def __len__(self):
        return len(self.itos)


class Multi30kDataset(Dataset):
    def __init__(self, split='train', src_vocab=None, tgt_vocab=None):
        self.split = split
        raw = load_dataset('bentrevett/multi30k')
        self.data = raw[split]
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_data = []
        self.tgt_data = []

    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self, min_freq=2):
        src_counter = Counter()
        tgt_counter = Counter()
        for item in self.data:
            src_counter.update(self.tokenize_de(item['de']))
            tgt_counter.update(self.tokenize_en(item['en']))

        self.src_vocab = Vocabulary()
        self.src_vocab.build_from_counter(src_counter, min_freq)

        self.tgt_vocab = Vocabulary()
        self.tgt_vocab.build_from_counter(tgt_counter, min_freq)

        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        assert self.src_vocab is not None and self.tgt_vocab is not None, \
            "Call build_vocab() first or pass src_vocab and tgt_vocab to constructor."

        sos_src = self.src_vocab.stoi['<sos>']
        eos_src = self.src_vocab.stoi['<eos>']
        sos_tgt = self.tgt_vocab.stoi['<sos>']
        eos_tgt = self.tgt_vocab.stoi['<eos>']

        self.src_data = []
        self.tgt_data = []

        for item in self.data:
            src_tokens = self.tokenize_de(item['de'])
            tgt_tokens = self.tokenize_en(item['en'])

            src_indices = [sos_src] + self.src_vocab.lookup_indices(src_tokens) + [eos_src]
            tgt_indices = [sos_tgt] + self.tgt_vocab.lookup_indices(tgt_tokens) + [eos_tgt]

            self.src_data.append(torch.tensor(src_indices, dtype=torch.long))
            self.tgt_data.append(torch.tensor(tgt_indices, dtype=torch.long))

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


def collate_fn(batch, pad_idx=1):
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=pad_idx
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=pad_idx
    )
    return src_padded, tgt_padded


def get_dataloader(dataset, batch_size=128, shuffle=True, pad_idx=1):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, pad_idx=pad_idx),
    )


def build_datasets(min_freq=2):
    train_dataset = Multi30kDataset(split='train')
    src_vocab, tgt_vocab = train_dataset.build_vocab(min_freq=min_freq)
    train_dataset.process_data()

    val_dataset = Multi30kDataset(split='validation', src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    val_dataset.process_data()

    test_dataset = Multi30kDataset(split='test', src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    test_dataset.process_data()

    return train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab