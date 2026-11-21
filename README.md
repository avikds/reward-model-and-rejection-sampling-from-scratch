# Reward Model and Rejection Sampling from Scratch

Rejection sampling is the simplest way to use a reward model to improve a language model: generate several completions for each prompt, score them with the reward model, keep the best one, and fine-tune the model on what was kept. This project builds that whole loop in PyTorch on a task small enough to train on a CPU in seconds. The task is to write a four-letter string backwards, inside a chat template made of two marker characters. The demonstration data is mediocre on purpose: its writers copy more letters than they reverse. You write a tiny causal transformer, instruction-tune it with a loss that counts only the completion tokens, and see that it reproduces the data's mix of right and wrong letters. Then you build preference pairs (two answers to one prompt, with a label for the better one), put a linear head with one output on the same transformer body to get a reward model, and train it with the pairwise Bradley-Terry loss. Last, you sample N completions per prompt, keep the one with the highest reward (best-of-N), fine-tune the generator on the kept answers, and measure the gain with the task's known scoring rule, which neither model ever sees.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** gold_score
- [x] **2.** make_demos
- [x] **3.** encode_example
- [x] **4.** TinyLM
- [x] **5.** sft_loss
- [x] **6.** instruction_tune
- [x] **7.** make_preference_pairs
- [x] **8.** RewardModel
- [x] **9.** pairwise_loss
- [x] **10.** train_reward_model
- [x] **11.** sample_completions
- [x] **12.** best_of_n
- [x] **13.** mean_gold_score
- [x] **14.** rejection_sampling

## Results

```
2087 demonstrations (average gold score 0.466), 2101 preference pairs, 306 prompts for rejection sampling, 196 held-out prompts

[1] instruction tuning: loss 2.497 -> 0.803
    generator gold score: sampled 0.453, greedy 0.385
    samples for 'abcd': ['abca', 'abbd', 'acca', 'dbcd', 'acbd', 'abbd', 'abcd', 'acca']

[2] reward model: loss 0.710 -> 0.043, accuracy on 295 held-out pairs 0.980
    rewards for 'abcd': dcba +10.85, dcbb +8.56, abcd -5.97, hhhh +6.26
    ('dcba' is correct, 'dcbb' has one wrong letter, 'abcd' is a copy, 'hhhh' is unlike anything in the preference data)

[3] best-of-N with the reward model (gold score of the selected answer)
    N =  1: 0.443
    N =  2: 0.574
    N =  4: 0.653
    N =  8: 0.716
    N = 16: 0.790

[4] rejection sampling (gold score on held-out prompts)
    kept data: best of 8 0.725, control with no selection (n = 1) 0.427
    instruction-tuned generator    sampled 0.453   greedy 0.385
    fine-tuned on best of 8        sampled 0.709   greedy 0.810
    control, fine-tuned on n = 1   sampled 0.436   greedy 0.415
    greedy answers for 'abcd', 'hgfe', 'cafe':
    instruction-tuned generator    ['abcd', 'hffe', 'caae']
    fine-tuned on best of 8        ['acba', 'efgh', 'efac']
    control, fine-tuned on n = 1   ['abba', 'egfe', 'caae']
```
