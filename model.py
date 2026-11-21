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

# Step 5 - sft_loss
def sft_loss(logits, ids, completion_mask):
    """Mean next-token negative log-likelihood over the completion tokens only."""

    # Shift logits and token IDs so each position predicts the next token.
    shift_logits = logits[:, :-1, :]
    shift_ids = ids[:, 1:]

    # Shift the completion mask to align with the prediction targets.
    shift_mask = completion_mask[:, 1:].to(
        device=logits.device,
        dtype=logits.dtype
    )

    # Calculate cross-entropy loss independently for each token.
    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_ids.reshape(-1),
        reduction="none"
    )

    # Restore the batch and sequence dimensions.
    token_losses = token_losses.view_as(shift_ids)

    # Count loss only for completion tokens.
    masked_loss = token_losses * shift_mask

    # Normalize by the total number of completion tokens in the batch.
    loss = masked_loss.sum() / shift_mask.sum()

    return loss

# Step 6 - instruction_tune
def instruction_tune(model, examples, steps, batch_size=32, lr=3e-3, seed=0):
    """AdamW on random batches of encoded examples with sft_loss; returns each step's loss as a float."""

    # Encode every demonstration example exactly once.
    encoded = [
        encode_example(prompt, completion)
        for prompt, completion in examples
    ]

    # Stack token IDs and completion masks into tensors.
    ids = torch.tensor(
        [item[0] for item in encoded],
        dtype=torch.long
    )

    masks = torch.tensor(
        [item[1] for item in encoded],
        dtype=torch.long
    )

    # Create a dedicated random-number generator for reproducible batches.
    g = torch.Generator().manual_seed(seed)

    # Initialize the AdamW optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr
    )

    # Set the model to training mode.
    model.train()

    losses = []

    for _ in range(steps):

        # Randomly sample a batch of demonstration examples.
        ix = torch.randint(
            0,
            len(examples),
            (batch_size,),
            generator=g
        )

        # Compute next-token logits for the selected examples.
        logits = model(ids[ix])

        # Calculate loss only on completion tokens.
        loss = sft_loss(
            logits,
            ids[ix],
            masks[ix]
        )

        # Reset gradients from the previous optimization step.
        optimizer.zero_grad()

        # Backpropagate the loss.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        # Record the current training loss as a Python float.
        losses.append(loss.item())

    return losses

# Step 7 - make_preference_pairs
def make_preference_pairs(n, seed, length=4, p_copy=0.6, p_typo=0.05):
    """n tuples (prompt, chosen, rejected); ties in gold score are discarded."""

    # Create one seeded random-number generator.
    rng = random.Random(seed)

    pairs = []

    # Continue generating until exactly n valid pairs are collected.
    while len(pairs) < n:

        # Generate a random prompt of the specified length.
        prompt = "".join(
            LETTERS[rng.randrange(len(LETTERS))]
            for _ in range(length)
        )

        # Generate two independent answers to the same prompt.
        a = write_answer(prompt, rng, p_copy, p_typo)
        b = write_answer(prompt, rng, p_copy, p_typo)

        # Evaluate both answers using the ground-truth scoring rule.
        score_a = gold_score(prompt, a)
        score_b = gold_score(prompt, b)

        # Discard pairs where both answers have equal scores.
        if score_a == score_b:
            continue

        # Place the better answer first.
        if score_a > score_b:
            pairs.append((prompt, a, b))
        else:
            pairs.append((prompt, b, a))

    return pairs

# Step 8 - RewardModel
class RewardModel(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_head=2, n_layer=2, max_len=16):
        """self.body: a TinyLM. self.head: a linear layer from d_model to one number."""

        super().__init__()

        # Create a new transformer backbone with its own independent weights.
        self.body = TinyLM(
            vocab_size,
            d_model,
            n_head,
            n_layer,
            max_len
        )

        # Predict one scalar reward from the final hidden state.
        self.head = nn.Linear(d_model, 1)

    def forward(self, ids):
        """(B, T) token ids -> (B,) rewards, read from the hidden state at the last position."""

        # Extract the hidden states from the transformer backbone.
        hidden_states = self.body.hidden(ids)

        # Select the hidden state at the final sequence position.
        last_hidden = hidden_states[:, -1, :]

        # Convert the final hidden state into a scalar reward.
        rewards = self.head(last_hidden)

        # Remove the final singleton dimension to obtain shape (B,).
        return rewards.squeeze(-1)

    def score(self, prompts, completions):
        """Lists of strings -> (len(prompts),) rewards, with gradients."""

        # Encode every prompt-completion pair using the existing chat template.
        encoded = [
            encode_example(prompt, completion)[0]
            for prompt, completion in zip(prompts, completions)
        ]

        # Stack token IDs into a single batch tensor.
        # Keep the tensor on the same device as the reward model.
        device = next(self.parameters()).device

        ids = torch.tensor(
            encoded,
            dtype=torch.long,
            device=device
        )

        # Return reward scores without disabling gradient computation.
        return self.forward(ids)

# Step 9 - pairwise_loss
def pairwise_loss(r_chosen, r_rejected):
    """Mean of -log sigmoid(r_chosen - r_rejected)."""

    # Calculate the reward difference for each preference pair.
    reward_diff = r_chosen - r_rejected

    # Use softplus for numerically stable computation:
    # -log(sigmoid(x)) = softplus(-x).
    losses = F.softplus(-reward_diff)

    # Return the mean loss across the batch.
    return losses.mean()


def pairwise_accuracy(r_chosen, r_rejected):
    """Fraction of pairs with r_chosen > r_rejected, as a Python float."""

    # A pair is correct only when the chosen reward is strictly higher.
    correct = (r_chosen > r_rejected).float()

    # Return the fraction of correctly ranked pairs as a Python float.
    return correct.mean().item()

# Step 10 - train_reward_model
def train_reward_model(rm, pairs, steps, batch_size=64, lr=5e-3, seed=0):
    """AdamW on random batches of pairs with pairwise_loss; returns each step's loss as a float."""

    # Create a dedicated random-number generator for reproducible batch sampling.
    g = torch.Generator().manual_seed(seed)

    # Initialize the AdamW optimizer.
    optimizer = torch.optim.AdamW(
        rm.parameters(),
        lr=lr
    )

    # Set the reward model to training mode.
    rm.train()

    losses = []

    for _ in range(steps):

        # Randomly sample preference-pair indices for the current batch.
        ix = torch.randint(
            0,
            len(pairs),
            (batch_size,),
            generator=g
        )

        # Retrieve the selected preference pairs.
        batch = [pairs[i] for i in ix.tolist()]

        # Separate prompts, chosen completions, and rejected completions.
        prompts = [p for p, c, r in batch]
        chosen = [c for p, c, r in batch]
        rejected = [r for p, c, r in batch]

        # Compute reward scores for both sets of completions.
        # Gradients must remain enabled for reward model training.
        r_chosen = rm.score(prompts, chosen)
        r_rejected = rm.score(prompts, rejected)

        # Calculate the Bradley-Terry pairwise loss.
        loss = pairwise_loss(r_chosen, r_rejected)

        # Reset gradients before backpropagation.
        optimizer.zero_grad()

        # Calculate gradients.
        loss.backward()

        # Update reward model parameters.
        optimizer.step()

        # Record the training loss as a Python float.
        losses.append(loss.item())

    return losses

# Step 11 - sample_completions
def sample_completions(model, prompt, n, temperature=1.0, generator=None):
    """n completions of len(prompt) tokens each, sampled token by token; temperature 0 is greedy."""

    # Switch the model to evaluation mode.
    model.eval()

    # Determine the model's device.
    # Parameter-free models should also be supported.
    parameter = next(model.parameters(), None)

    if parameter is not None:
        device = parameter.device
    else:
        buffer = next(model.buffers(), None)
        device = buffer.device if buffer is not None else torch.device("cpu")

    # Encode the prompt using the existing chat template.
    prompt_ids = encode_example(prompt)[0]

    # Create n identical copies of the encoded prompt.
    ids = torch.tensor(
        [prompt_ids] * n,
        dtype=torch.long,
        device=device
    )

    # Generate exactly len(prompt) new tokens without computing gradients.
    with torch.no_grad():
        for _ in range(len(prompt)):

            # Run the model on the complete sequences generated so far.
            # Only the logits at the last position are needed.
            logits = model(ids)[:, -1, :]

            if temperature == 0:
                # Greedy decoding: select the token with the highest logit.
                next_ids = logits.argmax(dim=-1, keepdim=True)

            else:
                # Apply temperature scaling before calculating probabilities.
                probs = F.softmax(logits / temperature, dim=-1)

                # Sample one new token independently for each sequence.
                next_ids = torch.multinomial(
                    probs,
                    num_samples=1,
                    generator=generator
                )

            # Append the sampled tokens so they become input to the next step.
            ids = torch.cat([ids, next_ids], dim=1)

    # Extract only the generated tokens, excluding the original prompt.
    completion_ids = ids[:, len(prompt_ids):]

    # Convert token IDs back into character strings.
    completions = [
        "".join(CHARS[token_id] for token_id in row.tolist())
        for row in completion_ids
    ]

    return completions

# Step 12 - best_of_n
def best_of_n(model, rm, prompt, n, temperature=1.0, generator=None):
    """Sample n completions, score them with rm, return (best, completions, scores)."""

    # Generate n candidate completions using the language model.
    completions = sample_completions(
        model,
        prompt,
        n,
        temperature,
        generator
    )

    # Set the reward model to evaluation mode.
    rm.eval()

    # Score all candidate completions without computing gradients.
    with torch.no_grad():
        scores = rm.score(
            [prompt] * n,
            completions
        )

    # Find the index of the highest reward.
    # torch.argmax returns the first index in case of a tie.
    best_idx = torch.argmax(scores).item()

    # Select the completion with the highest predicted reward.
    best = completions[best_idx]

    return best, completions, scores

# Step 13 - mean_gold_score
def mean_gold_score(model, prompts, n=4, temperature=1.0, seed=0):
    """Average gold score over n sampled completions for each prompt, as a Python float."""

    # Create one seeded generator for the entire evaluation.
    g = torch.Generator().manual_seed(seed)

    gold_scores = []

    # Process prompts in their original order.
    for prompt in prompts:

        # Sample n completions for the current prompt.
        completions = sample_completions(
            model,
            prompt,
            n,
            temperature,
            g
        )

        # Evaluate every completion using the ground-truth scoring rule.
        for completion in completions:
            score = gold_score(prompt, completion)
            gold_scores.append(score)

    # Return the average gold score as a Python float.
    return float(sum(gold_scores) / len(gold_scores))

# Step 14 - rejection_sampling (not yet solved)
# TODO: implement

