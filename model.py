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

# Step 2 - make_demos (not yet solved)
# TODO: implement

# Step 3 - encode_example (not yet solved)
# TODO: implement

# Step 4 - TinyLM (not yet solved)
# TODO: implement

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

