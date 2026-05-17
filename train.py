import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from dataset import build_datasets, get_dataloader


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            smooth_dist = torch.full_like(logits, self.smoothing / (self.vocab_size - 1))
            smooth_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            smooth_dist[target == self.pad_idx] = 0.0

        log_probs    = F.log_softmax(logits, dim=-1)
        loss         = -(smooth_dist * log_probs).sum(dim=-1)
        non_pad_mask = target != self.pad_idx
        return loss[non_pad_mask].mean()


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    log_grad_norms: bool = False,
) -> float:
    model.train() if is_train else model.eval()

    pad_idx      = model.pad_idx
    vocab_size   = model.output_projection.out_features
    total_loss   = 0.0
    total_tokens = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_idx, (src, tgt) in enumerate(tqdm(data_iter, desc=f"{'Train' if is_train else 'Val'} epoch {epoch_num}")):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx)

            logits = model(src, tgt_input, src_mask, tgt_mask)

            logits_flat = logits.reshape(-1, vocab_size)
            tgt_flat    = tgt_output.reshape(-1)

            loss = loss_fn(logits_flat, tgt_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if log_grad_norms and wandb.run is not None:
                    global_step = epoch_num * len(data_iter) + batch_idx
                    if global_step < 1000:
                        for name, param in model.named_parameters():
                            if param.grad is not None and ('W_q' in name or 'W_k' in name):
                                wandb.log({
                                    f"grad_norm/{name}": param.grad.norm().item(),
                                    "global_step": global_step,
                                })

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                if wandb.run is not None:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr":   optimizer.param_groups[0]["lr"],
                    })

            non_pad_tokens = (tgt_output != pad_idx).sum().item()
            total_loss   += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens

    avg_loss = total_loss / max(total_tokens, 1)

    if wandb.run is not None:
        prefix = "train" if is_train else "val"
        wandb.log({f"{prefix}/epoch_loss": avg_loss, "epoch": epoch_num})

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    pad_idx = model.pad_idx

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys     = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys       = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    model.eval()

    pad_idx = tgt_vocab.stoi['<pad>']
    sos_idx = tgt_vocab.stoi['<sos>']
    eos_idx = tgt_vocab.stoi['<eos>']
    src_pad = model.pad_idx

    hypotheses = []
    references = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval"):
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i    = src[i].unsqueeze(0)
                src_mask = make_src_mask(src_i, src_pad)

                pred_ids = greedy_decode(
                    model, src_i, src_mask, max_len,
                    start_symbol=sos_idx,
                    end_symbol=eos_idx,
                    device=device,
                ).squeeze(0).tolist()

                pred_words = []
                for idx in pred_ids[1:]:
                    if idx == eos_idx:
                        break
                    if idx not in (sos_idx, pad_idx):
                        pred_words.append(tgt_vocab.lookup_token(idx))

                gold_ids   = tgt[i].tolist()
                gold_words = []
                for idx in gold_ids[1:]:
                    if idx == eos_idx:
                        break
                    if idx not in (sos_idx, pad_idx):
                        gold_words.append(tgt_vocab.lookup_token(idx))

                hypotheses.append(' '.join(pred_words))
                references.append(' '.join(gold_words))

    import sacrebleu
    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
        tokenize='13a',
        lowercase=True,
    )
    return bleu.score


# ══════════════════════════════════════════════════════════════════════
#  CONFIDENCE LOGGING (experiment 5 — label smoothing)
# ══════════════════════════════════════════════════════════════════════

def log_prediction_confidence(model, data_iter, pad_idx, vocab_size, device, epoch_num):
    model.eval()
    total_conf  = 0.0
    total_count = 0

    with torch.no_grad():
        for src, tgt in data_iter:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx)

            logits  = model(src, tgt_input, src_mask, tgt_mask)
            probs   = F.softmax(logits, dim=-1)

            flat_probs  = probs.reshape(-1, vocab_size)
            flat_target = tgt_output.reshape(-1)
            non_pad     = flat_target != pad_idx

            correct_probs = flat_probs[non_pad].gather(1, flat_target[non_pad].unsqueeze(1)).squeeze(1)
            total_conf  += correct_probs.sum().item()
            total_count += non_pad.sum().item()

    avg_conf = total_conf / max(total_count, 1)
    if wandb.run is not None:
        wandb.log({"val/prediction_confidence": avg_conf, "epoch": epoch_num})
    return avg_conf


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    model_config = {
        'src_vocab_size': model.src_embedding.num_embeddings,
        'tgt_vocab_size': model.tgt_embedding.num_embeddings,
        'd_model':        model.d_model,
        'N':              len(model.encoder.layers),
        'num_heads':      model.encoder.layers[0].self_attn.num_heads,
        'd_ff':           model.encoder.layers[0].ffn.linear1.out_features,
        'dropout':        model.encoder.layers[0].dropout.p,
        'pad_idx':        model.pad_idx,
        'pe_type':        model.pe_type,
        'scale':          model.scale,
    }
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'model_config':         model_config,
        'src_vocab':            model.src_vocab,
        'tgt_vocab':            model.tgt_vocab,
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if 'src_vocab' in checkpoint:
        model.src_vocab = checkpoint['src_vocab']
    if 'tgt_vocab' in checkpoint:
        model.tgt_vocab = checkpoint['tgt_vocab']

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch']


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(args) -> None:
    config = {
        'd_model':       args.d_model,
        'N':             args.N,
        'num_heads':     args.num_heads,
        'd_ff':          args.d_ff,
        'dropout':       args.dropout,
        'batch_size':    args.batch_size,
        'num_epochs':    args.epochs,
        'warmup_steps':  args.warmup_steps,
        'min_freq':      2,
        'pad_idx':       1,
        'smoothing':     args.smoothing,
        'max_len':       100,
        'scheduler':     args.scheduler,
        'pe_type':       args.pe_type,
        'attn_scale':    not args.no_attn_scale,
    }

    wandb.init(project="da6401-a3", name=args.run_name, config=config)
    cfg = wandb.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = build_datasets(min_freq=cfg.min_freq)
    print(f"src vocab: {len(src_vocab)}  tgt vocab: {len(tgt_vocab)}")

    train_loader = get_dataloader(train_dataset, batch_size=cfg.batch_size, shuffle=True,  pad_idx=cfg.pad_idx)
    val_loader   = get_dataloader(val_dataset,   batch_size=cfg.batch_size, shuffle=False, pad_idx=cfg.pad_idx)
    test_loader  = get_dataloader(test_dataset,  batch_size=cfg.batch_size, shuffle=False, pad_idx=cfg.pad_idx)

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        pad_idx=cfg.pad_idx,
        pe_type=cfg.pe_type,
        scale=cfg.attn_scale,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
    ).to(device)

    if cfg.scheduler == 'noam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = None

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab), pad_idx=cfg.pad_idx, smoothing=cfg.smoothing
    )

    log_grad_norms = args.no_attn_scale

    best_bleu      = -1.0
    checkpoint_path = f"best_checkpoint_{args.run_name}.pt"

    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
            log_grad_norms=log_grad_norms,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )
        val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=device, max_len=cfg.max_len)

        if cfg.smoothing == 0.0 or True:
            log_prediction_confidence(
                model, val_loader, cfg.pad_idx, len(tgt_vocab), device, epoch
            )

        print(f"Epoch {epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_bleu={val_bleu:.2f}")

        wandb.log({
            "val/bleu":       val_bleu,
            "val/epoch_loss": val_loss,
            "epoch":          epoch,
        })

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            save_checkpoint(model, optimizer, scheduler, epoch, path=checkpoint_path)
            print(f"  Saved best checkpoint (BLEU={best_bleu:.2f})")

    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=cfg.max_len)
    print(f"Test BLEU: {test_bleu:.2f}")
    wandb.log({"test/bleu": test_bleu})
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 3 - Transformer Training")

    parser.add_argument('--run_name',      type=str,   default='base_noam',    help='W&B run name')
    parser.add_argument('--scheduler',     type=str,   default='noam',         choices=['noam', 'fixed'], help='LR scheduler type')
    parser.add_argument('--lr',            type=float, default=1e-4,           help='Fixed LR (only used when --scheduler fixed)')
    parser.add_argument('--no_attn_scale', action='store_true',                help='Disable 1/sqrt(dk) scaling in attention')
    parser.add_argument('--pe_type',       type=str,   default='sinusoidal',   choices=['sinusoidal', 'learned'], help='Positional encoding type')
    parser.add_argument('--smoothing',     type=float, default=0.1,            help='Label smoothing epsilon')

    parser.add_argument('--d_model',       type=int,   default=256)
    parser.add_argument('--N',             type=int,   default=3)
    parser.add_argument('--num_heads',     type=int,   default=8)
    parser.add_argument('--d_ff',          type=int,   default=512)
    parser.add_argument('--dropout',       type=float, default=0.1)
    parser.add_argument('--batch_size',    type=int,   default=128)
    parser.add_argument('--epochs',        type=int,   default=20)
    parser.add_argument('--warmup_steps',  type=int,   default=4000)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training_experiment(args)