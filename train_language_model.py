"""Train an attention-free character language model.

This script adds a real-language evaluation path to the repository. It is
intended as a lightweight bridge between the existing synthetic benchmarks and
natural-language modelling experiments.

The model is deliberately simple and fully attention-free:

- token embedding + learned positional embedding
- causal depthwise convolutions for local and dilated temporal mixing
- pointwise gated channel mixing
- residual feed-forward layers
- tied input/output embeddings

It does not claim state-of-the-art performance. Its purpose is to make it easy
to run repeatable char-level language-modelling smoke tests on a plain text
corpus before scaling the discovered architecture family to larger benchmarks.

Example:
    python train_language_model.py --data-path data/tiny.txt --epochs 5
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    data_path: str
    output_dir: str = "runs/language_model"
    seq_len: int = 256
    stride: int = 256
    batch_size: int = 32
    d_model: int = 256
    num_layers: int = 8
    kernel_size: int = 5
    dropout: float = 0.1
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    valid_fraction: float = 0.05
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_every_epoch: bool = False
    early_stopping_patience: int = 5
    dry_run: bool = False


class CharacterTokenizer:
    """Minimal character tokenizer with a fixed vocabulary."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        if not chars:
            raise ValueError("Cannot build a tokenizer from an empty corpus.")
        self.stoi: Dict[str, int] = {ch: idx for idx, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {idx: ch for ch, idx in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(idx)] for idx in ids)

    def to_json(self) -> Dict[str, object]:
        return {"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}


class LanguageModelingDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Overlapping fixed-length next-token-prediction windows."""

    def __init__(self, token_ids: List[int], seq_len: int, stride: int) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2.")
        if stride < 1:
            raise ValueError("stride must be at least 1.")
        if len(token_ids) <= seq_len:
            raise ValueError(
                f"Corpus split has {len(token_ids)} tokens, but seq_len={seq_len}. "
                "Use a shorter sequence length or a larger corpus."
            )
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.starts = list(range(0, len(token_ids) - seq_len, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        start = self.starts[index]
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution with strict left padding.

    Unlike symmetric padding, this does not leak future tokens into the current
    hidden state. That matters for autoregressive language modelling.
    """

    def __init__(self, d_model: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=d_model,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        h = x.transpose(1, 2)
        h = F.pad(h, (self.left_padding, 0))
        h = self.conv(h)
        return h.transpose(1, 2)


class AttentionFreeBlock(nn.Module):
    """Causal attention-free mixer block."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.temporal = CausalDepthwiseConv1d(d_model, kernel_size, dilation)
        self.channel_gate = nn.Linear(d_model, 2 * d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = self.temporal(h)
        gate, value = self.channel_gate(h).chunk(2, dim=-1)
        h = torch.sigmoid(gate) * F.gelu(value)
        x = x + self.dropout(h)
        x = x + self.ffn(self.norm2(x))
        return x


class AttentionFreeLanguageModel(nn.Module):
    """Small char-level language model with no attention layers."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        dilations = [2 ** (idx % 6) for idx in range(num_layers)]
        self.blocks = nn.ModuleList(
            [
                AttentionFreeBlock(d_model, kernel_size, dilation, dropout)
                for dilation in dilations
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, x: Tensor) -> Tensor:
        batch_size, length = x.shape
        if length > self.seq_len:
            raise ValueError(f"Input length {length} exceeds model seq_len {self.seq_len}.")
        positions = torch.arange(length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.drop(h)
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_norm(h))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_text(text: str, valid_fraction: float) -> Tuple[str, str]:
    if not 0.0 < valid_fraction < 0.5:
        raise ValueError("valid_fraction must be between 0 and 0.5.")
    split_index = int(len(text) * (1.0 - valid_fraction))
    return text[:split_index], text[split_index:]


def collate_batch(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    xs, ys = zip(*batch)
    return torch.stack(list(xs), dim=0), torch.stack(list(ys), dim=0)


def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = compute_loss(logits, y)
        total_loss += float(loss.item()) * y.numel()
        total_tokens += int(y.numel())
    mean_loss = total_loss / max(total_tokens, 1)
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20.0))}


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = Path(config.data_path).read_text(encoding="utf-8")
    train_text, valid_text = split_text(text, config.valid_fraction)
    tokenizer = CharacterTokenizer(text)

    train_dataset = LanguageModelingDataset(
        tokenizer.encode(train_text), config.seq_len, config.stride
    )
    valid_dataset = LanguageModelingDataset(
        tokenizer.encode(valid_text), config.seq_len, config.stride
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_batch,
    )

    model = AttentionFreeLanguageModel(
        vocab_size=tokenizer.vocab_size,
        seq_len=config.seq_len,
        d_model=config.d_model,
        num_layers=config.num_layers,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    metadata = {
        "config": asdict(config),
        "vocab_size": tokenizer.vocab_size,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "tokenizer": tokenizer.to_json(),
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    best_valid_loss = float("inf")
    patience_counter = 0
    if config.dry_run:
        if len(train_loader) == 0:
            raise ValueError(
                "Dry-run requires at least one training batch. "
                "Use a larger corpus, smaller seq_len, or smaller batch_size."
            )
        model.train()
        x, y = next(iter(train_loader))
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(x)
            loss = compute_loss(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer); scaler.update()
        print(f"dry_run_ok loss={loss.item():.4f} grad_norm={float(grad_norm):.4f}")
        return

    for epoch in range(1, config.epochs + 1):
        model.train()
        start_time = time.time()
        total_loss = 0.0
        total_tokens = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x)
                loss = compute_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * y.numel()
            total_tokens += int(y.numel())

            if step % 50 == 0:
                mean_train_loss = total_loss / max(total_tokens, 1)
                print(
                    f"epoch={epoch} step={step} "
                    f"train_loss={mean_train_loss:.4f} "
                    f"train_ppl={math.exp(min(mean_train_loss, 20.0)):.2f} grad_norm={float(grad_norm):.4f}"
                )

        train_loss = total_loss / max(total_tokens, 1)
        valid_metrics = evaluate(model, valid_loader, device)
        elapsed = time.time() - start_time
        print(
            f"epoch={epoch} done elapsed={elapsed:.1f}s "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_ppl={valid_metrics['perplexity']:.2f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "valid_metrics": valid_metrics,
            "metadata": metadata,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if config.save_every_epoch:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            patience_counter = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"early_stopping epoch={epoch} patience={config.early_stopping_patience}")
                break


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="Path to a UTF-8 text corpus.")
    parser.add_argument("--output-dir", default="runs/language_model")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
