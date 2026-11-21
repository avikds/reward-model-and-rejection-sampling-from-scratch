"""
Reward Model and Rejection Sampling from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - gold_score
LETTERS = "abcdefgh"

def gold_score(prompt, completion):
    """Fraction of positions where `completion` matches `prompt` written backwards."""

    # Reverse the prompt to obtain the target completion.
    target = prompt[::-1]

    # Count matching characters at corresponding positions.
    matching_positions = sum(
        target_char == completion_char
        for target_char, completion_char in zip(target, completion)
    )

    # Normalize by the longer string length.
    # This accounts for missing or extra characters.
    denominator = max(len(target), len(completion))

    return matching_positions / denominator

# Step 2 - make_demos
import random


def write_answer(prompt, rng, p_copy=0.6, p_typo=0.05):
    """One answer from a writer who, at each position, copies the prompt's letter with probability p_copy and mistypes with probability p_typo."""

    answer = []

    for i in range(len(prompt)):
        # First decide whether to copy the original letter or reverse it.
        if rng.random() < p_copy:
            char = prompt[i]
        else:
            char = prompt[len(prompt) - 1 - i]

        # Independently decide whether to introduce a typo.
        if rng.random() < p_typo:
            char = LETTERS[rng.randrange(len(LETTERS))]

        answer.append(char)

    return "".join(answer)


def make_demos(n, seed, length=4, p_copy=0.6, p_typo=0.05):
    """n (prompt, completion) pairs from random.Random(seed)."""

    # Create one seeded random-number generator for reproducibility.
    rng = random.Random(seed)

    demos = []

    for _ in range(n):
        # Generate a random prompt using the available LETTERS constant.
        prompt = "".join(
            LETTERS[rng.randrange(len(LETTERS))]
            for _ in range(length)
        )

        # Generate the corresponding demonstration completion.
        completion = write_answer(prompt, rng, p_copy, p_typo)

        demos.append((prompt, completion))

    return demos

# Step 3 - encode_example
# Define the character-level vocabulary.
CHARS = LETTERS + "#:"

# Map each character to its corresponding token ID.
STOI = {char: idx for idx, char in enumerate(CHARS)}

# Total number of tokens in the vocabulary.
VOCAB_SIZE = len(CHARS)


def encode_example(prompt, completion=""):
    """Template text '#' + prompt + ':' + completion -> (ids, completion_mask)."""

    # Construct the chat template.
    text = "#" + prompt + ":" + completion

    # Convert every character into its corresponding token ID.
    ids = [STOI[char] for char in text]

    # Prompt and chat markers are masked with 0.
    # Only completion tokens are marked with 1.
    completion_mask = (
        [0] * (len(prompt) + 2)
        + [1] * len(completion)
    )

    return ids, completion_mask

# Step 4 - TinyLM
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal attention mask."""

    def __init__(self, d_model, n_head):
        super().__init__()

        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head.")

        self.n_head = n_head
        self.head_size = d_model // n_head

        # Project input into query, key, and value vectors.
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # Output projection after combining attention heads.
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape

        # Compute query, key, and value projections.
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape into multiple attention heads.
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Calculate scaled dot-product attention scores.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)

        # Create a causal mask that prevents attending to future tokens.
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1
        )

        # Mask future positions before applying softmax.
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Normalize attention scores into probabilities.
        weights = F.softmax(scores, dim=-1)

        # Apply attention weights to the value vectors.
        out = weights @ v

        # Combine the multiple attention heads.
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Apply the final output projection.
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-normalization Transformer block with attention and MLP."""

    def __init__(self, d_model, n_head):
        super().__init__()

        # Layer normalization before attention.
        self.ln1 = nn.LayerNorm(d_model)

        # Multi-head causal self-attention.
        self.attention = CausalSelfAttention(d_model, n_head)

        # Layer normalization before the feed-forward network.
        self.ln2 = nn.LayerNorm(d_model)

        # Feed-forward MLP with GELU activation.
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # Attention sublayer with a residual connection.
        x = x + self.attention(self.ln1(x))

        # MLP sublayer with a residual connection.
        x = x + self.mlp(self.ln2(x))

        return x


class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_head=2, n_layer=2, max_len=16):
        """Token and position embeddings, n_layer causal pre-norm blocks, a final LayerNorm, and self.lm_head."""

        super().__init__()

        self.max_len = max_len

        # Token embeddings convert token IDs into vector representations.
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings provide information about token positions.
        self.position_embedding = nn.Embedding(max_len, d_model)

        # Stack multiple causal Transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head)
            for _ in range(n_layer)
        ])

        # Final normalization of the hidden states.
        self.final_norm = nn.LayerNorm(d_model)

        # Language-model head predicts the next-token logits.
        self.lm_head = nn.Linear(d_model, vocab_size)

    def hidden(self, ids):
        """(B, T) token ids -> (B, T, d_model) hidden states; position t sees tokens 0..t only."""

        if ids.ndim != 2:
            raise ValueError("ids must have shape (B, T).")

        B, T = ids.shape

        if T > self.max_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_len={self.max_len}."
            )

        # Generate position indices for the input sequence.
        positions = torch.arange(T, device=ids.device)

        # Combine token embeddings with positional embeddings.
        x = (
            self.token_embedding(ids)
            + self.position_embedding(positions).unsqueeze(0)
        )

        # Pass the embeddings through all Transformer blocks.
        for block in self.blocks:
            x = block(x)

        # Apply final LayerNorm to obtain hidden states.
        return self.final_norm(x)

    def forward(self, ids):
        """(B, T) token ids -> (B, T, vocab_size) next-token logits."""

        # Convert input token IDs into hidden representations.
        hidden_states = self.hidden(ids)

        # Project hidden states into vocabulary-sized logits.
        logits = self.lm_head(hidden_states)

        return logits

# Step 5 - sft_loss (not yet solved)
# TODO: implement

# Step 6 - instruction_tune (not yet solved)
# TODO: implement

# Step 7 - make_preference_pairs (not yet solved)
# TODO: implement

# Step 8 - RewardModel (not yet solved)
# TODO: implement

# Step 9 - pairwise_loss (not yet solved)
# TODO: implement

# Step 10 - train_reward_model (not yet solved)
# TODO: implement

# Step 11 - sample_completions (not yet solved)
# TODO: implement

# Step 12 - best_of_n (not yet solved)
# TODO: implement

# Step 13 - mean_gold_score (not yet solved)
# TODO: implement

# Step 14 - rejection_sampling (not yet solved)
# TODO: implement

