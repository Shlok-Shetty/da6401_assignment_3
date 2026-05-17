# sanity_check.py
from dataset import build_datasets, get_dataloader
from model import Transformer, make_src_mask, make_tgt_mask
import torch

train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = build_datasets()
print(f"src vocab: {len(src_vocab)}  tgt vocab: {len(tgt_vocab)}")
print(f"train size: {len(train_dataset)}  val: {len(val_dataset)}  test: {len(test_dataset)}")

train_loader = get_dataloader(train_dataset, batch_size=32)
src, tgt = next(iter(train_loader))
print(f"src shape: {src.shape}  tgt shape: {tgt.shape}")

model = Transformer(len(src_vocab), len(tgt_vocab), d_model=256, N=3, num_heads=8, d_ff=512)
src_mask = make_src_mask(src)
tgt_mask = make_tgt_mask(tgt[:, :-1])
out = model(src, tgt[:, :-1], src_mask, tgt_mask)
print(f"output shape: {out.shape}")
print("Sanity check passed")