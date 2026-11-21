# Reward Model and Rejection Sampling from Scratch

Rejection sampling is the simplest way to use a reward model to improve a language model: generate several completions for each prompt, score them with the reward model, keep the best one, and fine-tune the model on what was kept. This project builds that whole loop in PyTorch on a task small enough to train on a CPU in seconds. The task is to write a four-letter string backwards, inside a chat template made of two marker characters. The demonstration data is mediocre on purpose: its writers copy more letters than they reverse. You write a tiny causal transformer, instruction-tune it with a loss that counts only the completion tokens, and see that it reproduces the data's mix of right and wrong letters. Then you build preference pairs (two answers to one prompt, with a label for the better one), put a linear head with one output on the same transformer body to get a reward model, and train it with the pairwise Bradley-Terry loss. Last, you sample N completions per prompt, keep the one with the highest reward (best-of-N), fine-tune the generator on the kept answers, and measure the gain with the task's known scoring rule, which neither model ever sees.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** gold_score
- [ ] **2.** make_demos
- [ ] **3.** encode_example
- [ ] **4.** TinyLM
- [ ] **5.** sft_loss
- [ ] **6.** instruction_tune
- [ ] **7.** make_preference_pairs
- [ ] **8.** RewardModel
- [ ] **9.** pairwise_loss
- [ ] **10.** train_reward_model
- [ ] **11.** sample_completions
- [ ] **12.** best_of_n
- [ ] **13.** mean_gold_score
- [ ] **14.** rejection_sampling

---

Built on Deep-ML.
