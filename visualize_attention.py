import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import wandb

from model import Transformer, make_src_mask
from dataset import build_datasets


CHECKPOINT   = 'best_checkpoint_base_noam.pt'
SENTENCE     = "Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen."
WANDB_PROJECT = "da6401-a3"
WANDB_RUN     = "attention_head_visualization"


def load_model(checkpoint_path, src_vocab, tgt_vocab, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt['model_config']

    model = Transformer(
        src_vocab_size=cfg['src_vocab_size'],
        tgt_vocab_size=cfg['tgt_vocab_size'],
        d_model=cfg['d_model'],
        N=cfg['N'],
        num_heads=cfg['num_heads'],
        d_ff=cfg['d_ff'],
        dropout=cfg['dropout'],
        pad_idx=cfg['pad_idx'],
        pe_type=cfg.get('pe_type', 'sinusoidal'),
        scale=cfg.get('scale', True),
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        load_pretrained=False,
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


def get_encoder_attention_weights(model, src_tensor, src_mask):
    with torch.no_grad():
        x = model.pe(model.src_embedding(src_tensor) * (model.d_model ** 0.5))
        for layer in model.encoder.layers:
            normed = layer.norm1(x)
            layer.self_attn(normed, normed, normed, src_mask)
            x = layer(x, src_mask)
    last_layer = model.encoder.layers[-1]
    return last_layer.self_attn.attn_weights.squeeze(0).cpu()


def plot_head_heatmap(attn_weights, tokens, head_idx, ax):
    data = attn_weights[head_idx].numpy()
    im   = ax.imshow(data, cmap='viridis', aspect='auto', vmin=0, vmax=data.max())
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(tokens, fontsize=7)
    ax.set_title(f"Head {head_idx + 1}", fontsize=9)
    ax.set_xlabel("Key", fontsize=7)
    ax.set_ylabel("Query", fontsize=7)
    return im


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _, _, _, src_vocab, tgt_vocab = build_datasets()

    model, cfg = load_model(CHECKPOINT, src_vocab, tgt_vocab, device)
    num_heads   = cfg['num_heads']

    import spacy
    spacy_de = spacy.load('de_core_news_sm')
    tokens   = [tok.text.lower() for tok in spacy_de.tokenizer(SENTENCE)]
    tokens   = ['<sos>'] + tokens + ['<eos>']

    sos_idx  = src_vocab.stoi['<sos>']
    eos_idx  = src_vocab.stoi['<eos>']
    indices  = [sos_idx] + src_vocab.lookup_indices(tokens[1:-1]) + [eos_idx]

    src_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    src_mask   = make_src_mask(src_tensor, cfg['pad_idx'])

    attn_weights = get_encoder_attention_weights(model, src_tensor, src_mask)
    print(f"Attention weights shape: {attn_weights.shape}")

    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for h in range(num_heads):
        plot_head_heatmap(attn_weights, tokens, h, axes[h])

    plt.suptitle(f"Last Encoder Layer — All {num_heads} Attention Heads\n\"{SENTENCE}\"", fontsize=11)
    plt.tight_layout()
    plt.savefig("attention_heads.png", dpi=150, bbox_inches='tight')
    plt.close()

    wandb.log({"attention/all_heads": wandb.Image("attention_heads.png")})

    for h in range(num_heads):
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_head_heatmap(attn_weights, tokens, h, ax)
        plt.tight_layout()
        fname = f"attention_head_{h+1}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        wandb.log({f"attention/head_{h+1}": wandb.Image(fname)})
        print(f"Logged head {h+1}")

    print("\nAttention weight stats per head:")
    for h in range(num_heads):
        w = attn_weights[h]
        diag_mass  = w.diag().mean().item()
        max_pos    = w.mean(0).argmax().item()
        print(f"  Head {h+1}: diagonal_mass={diag_mass:.3f}  most_attended_key={tokens[max_pos]!r}")

    wandb.finish()
    print("Done. Check W&B for heatmaps.")


if __name__ == "__main__":
    main()