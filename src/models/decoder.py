"""LPN decoder p(y | x, z), ported from lpn's DecoderTransformer (prompts/PROJECT-SPEC-1.md section 2.3).

Token sequence (37 tokens for 4x4 grids), as in the original:
   0      rows_in   (shape embedding + input embedding)
   1      cols_in
   2..17  input cells    (color + position + input embedding)
  18      context z      (bias-free linear projection of z)
  19      rows_out  (shape embedding + output embedding)
  20      cols_out
  21..36  output cells   (color + *next*-position + output embedding)
Attention: input tokens (0-17) see every input token and the context token;
the output part (18-36) sees all input tokens and is causal within itself.
Logits (each through a parameter-free LayerNorm): output-rows from token 18,
output-cols from 19, and output cell k from token 20 + k. Output cell k's token
carries the position embedding of cell k + 1 ("next-position" embeddings), i.e.
of the cell it helps predict.

Simplifications that are exact for our always-4x4 grids: no padding masks, and
the original's "copy logits from the last non-padded column" step is the
identity. Output-shape tokens are teacher-forced to 4x4 (`nll` still scores
the model's shape predictions, as the original loss does).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.config import GRID_SIZE, NUM_COLORS, ModelConfig
from src.models.transformer import TransformerLayer, layer_norm

NUM_CELLS = GRID_SIZE * GRID_SIZE
SEQ_LEN = 2 * (2 + NUM_CELLS) + 1
INPUT_LEN = 2 + NUM_CELLS  # tokens before the context token
SHAPE_TARGET = GRID_SIZE - 1  # shape classes are size - 1


def attention_mask(device: torch.device | str | None = None) -> torch.Tensor:
    """(SEQ_LEN, SEQ_LEN) bool, True = may attend (the original's make_causal_pad_mask, unpadded)."""
    mask = torch.zeros(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=device)
    mask[:INPUT_LEN, :INPUT_LEN] = True  # input <-> input
    mask[:INPUT_LEN, INPUT_LEN] = True  # input -> context token
    mask[INPUT_LEN:, :INPUT_LEN] = True  # output part -> input
    n_out = SEQ_LEN - INPUT_LEN
    mask[INPUT_LEN:, INPUT_LEN:] = torch.ones(n_out, n_out, dtype=torch.bool, device=device).tril()
    return mask


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        dim = cfg.emb_dim
        self.context_embed = nn.Linear(cfg.latent_dim, dim, bias=False)
        self.pos_row_embed = nn.Embedding(GRID_SIZE, dim)
        self.pos_col_embed = nn.Embedding(GRID_SIZE, dim)
        self.grid_shapes_row_embed = nn.Embedding(GRID_SIZE, dim)
        self.grid_shapes_col_embed = nn.Embedding(GRID_SIZE, dim)
        self.colors_embed = nn.Embedding(NUM_COLORS, dim)
        self.input_output_embed = nn.Embedding(2, dim)
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(TransformerLayer(cfg) for _ in range(cfg.num_layers))
        self.shape_row_logits_proj = nn.Linear(dim, GRID_SIZE, bias=False)
        self.shape_col_logits_proj = nn.Linear(dim, GRID_SIZE, bias=False)
        self.grid_logits_proj = nn.Linear(dim, NUM_COLORS, bias=False)
        self.register_buffer("mask", attention_mask(), persistent=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        """Teacher-forced. x, y (..., 4, 4); z (..., d) ->
        (row_logits (..., 4), col_logits (..., 4), grid_logits (..., 16, 10)), float32."""
        lead = x.shape[:-2]
        dtype = self.context_embed.weight.dtype
        in_emb, out_emb = self.input_output_embed.weight
        pos = (self.pos_row_embed.weight[:, None] + self.pos_col_embed.weight[None, :]).flatten(0, 1)  # (16, D)
        next_pos = pos.roll(-1, dims=0)  # cell k gets cell k+1's position; the last wraps to (0, 0)
        full = torch.full((), SHAPE_TARGET, dtype=torch.long, device=x.device)
        rows, cols = self.grid_shapes_row_embed(full), self.grid_shapes_col_embed(full)

        tokens = torch.cat([
            torch.stack([rows + in_emb, cols + in_emb]).expand(*lead, -1, -1),
            self.colors_embed(x.flatten(-2)) + pos + in_emb,
            self.context_embed(z.to(dtype))[..., None, :],
            torch.stack([rows + out_emb, cols + out_emb]).expand(*lead, -1, -1),
            self.colors_embed(y.flatten(-2)) + next_pos + out_emb,
        ], dim=-2)
        h = self.embed_dropout(tokens).reshape(-1, SEQ_LEN, tokens.shape[-1])
        for layer in self.layers:
            h = layer(h, self.mask)
        h = h.reshape(*lead, SEQ_LEN, -1)
        row_logits = self.shape_row_logits_proj(layer_norm(h[..., INPUT_LEN, :]))
        col_logits = self.shape_col_logits_proj(layer_norm(h[..., INPUT_LEN + 1, :]))
        grid_logits = self.grid_logits_proj(layer_norm(h[..., INPUT_LEN + 2 : -1, :]))
        return row_logits.float(), col_logits.float(), grid_logits.float()

    def nll(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """The original per-pair loss: -log p(rows) - log p(cols) + mean over cells of -log p(cell).
        x, y (..., 4, 4); z (..., d) -> (...)."""
        row_logits, col_logits, grid_logits = self(x, y, z)
        shape_target = torch.full(row_logits.shape[:-1], SHAPE_TARGET, dtype=torch.long, device=x.device)
        shape_nll = (F.cross_entropy(row_logits.flatten(0, -2), shape_target.flatten(), reduction="none")
                     + F.cross_entropy(col_logits.flatten(0, -2), shape_target.flatten(), reduction="none"))
        cell_nll = F.cross_entropy(grid_logits.flatten(0, -2), y.flatten(), reduction="none")
        return shape_nll.view(shape_target.shape) + cell_nll.view(*y.shape[:-2], NUM_CELLS).mean(-1)

    @torch.no_grad()
    def generate(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Greedy autoregressive decode (shape tokens teacher-forced to 4x4). x (..., 4, 4), z (..., d) -> (..., 4, 4)."""
        y = torch.zeros_like(x).flatten(-2)
        for cell in range(NUM_CELLS):
            _, _, logits = self(x, y.unflatten(-1, (GRID_SIZE, GRID_SIZE)), z)
            y[..., cell] = logits[..., cell, :].argmax(-1)
        return y.unflatten(-1, (GRID_SIZE, GRID_SIZE))

    @torch.no_grad()
    def shape_correct(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Diagnostic: does the model predict a 4x4 output? (...) bool."""
        row_logits, col_logits, _ = self(x, torch.zeros_like(x), z)
        return (row_logits.argmax(-1) == SHAPE_TARGET) & (col_logits.argmax(-1) == SHAPE_TARGET)
