"""
Reward Model and Rejection Sampling from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Reward model and rejection sampling, end to end, on the string-reversal task.

Story: instruction-tune a tiny generator on demonstrations of mixed quality, train a reward model on
preference pairs, then use the reward model twice: at inference time (best-of-N) and to build new
training data (rejection sampling). Every stage is measured with the gold score on held-out prompts
that none of the training data contains. Takes about a minute on a CPU.
"""
import torch


def main():
    # ---- 0. Data. The held-out prompts are removed from everything the models train on ----
    held = set(p for p, _ in make_demos(200, seed=999))
    eval_prompts = sorted(held)
    demos = [d for d in make_demos(2200, seed=0) if d[0] not in held]
    pairs = [t for t in make_preference_pairs(2200, seed=1) if t[0] not in held]
    test_pairs = [t for t in make_preference_pairs(6000, seed=2) if t[0] in held]
    rs_prompts = [p for p, _ in make_demos(320, seed=7) if p not in held]
    demo_gold = sum(gold_score(p, c) for p, c in demos) / len(demos)
    print(f"{len(demos)} demonstrations (average gold score {demo_gold:.3f}), {len(pairs)} preference pairs, "
          f"{len(rs_prompts)} prompts for rejection sampling, {len(eval_prompts)} held-out prompts")

    # ---- 1. Instruction tuning ----
    torch.manual_seed(0)
    gen = TinyLM(VOCAB_SIZE)
    losses = instruction_tune(gen, demos, steps=400)
    print(f"\n[1] instruction tuning: loss {losses[0]:.3f} -> {sum(losses[-30:]) / 30:.3f}")
    base_sampled = mean_gold_score(gen, eval_prompts)
    base_greedy = mean_gold_score(gen, eval_prompts, n=1, temperature=0)
    print(f"    generator gold score: sampled {base_sampled:.3f}, greedy {base_greedy:.3f}")
    print("    samples for 'abcd':", sample_completions(gen, "abcd", 8, generator=torch.Generator().manual_seed(0)))

    # ---- 2. The reward model ----
    torch.manual_seed(1)
    rm = RewardModel(VOCAB_SIZE)
    losses = train_reward_model(rm, pairs, steps=400)
    rm.eval()
    P, C, R = zip(*test_pairs)
    with torch.no_grad():
        acc = pairwise_accuracy(rm.score(P, C), rm.score(P, R))
        r = rm.score(["abcd"] * 4, ["dcba", "dcbb", "abcd", "hhhh"]).tolist()
    print(f"\n[2] reward model: loss {losses[0]:.3f} -> {sum(losses[-30:]) / 30:.3f}, "
          f"accuracy on {len(test_pairs)} held-out pairs {acc:.3f}")
    print("    rewards for 'abcd': " + ", ".join(f"{c} {x:+.2f}" for c, x in zip(["dcba", "dcbb", "abcd", "hhhh"], r)))
    print("    ('dcba' is correct, 'dcbb' has one wrong letter, 'abcd' is a copy, 'hhhh' is unlike anything in the preference data)")

    # ---- 3. Best-of-N at inference time: better answers, same weights ----
    print("\n[3] best-of-N with the reward model (gold score of the selected answer)")
    for n in (1, 2, 4, 8, 16):
        g = torch.Generator().manual_seed(n)
        picks = [best_of_n(gen, rm, p, n, generator=g)[0] for p in eval_prompts]
        print(f"    N = {n:>2}: {sum(gold_score(p, c) for p, c in zip(eval_prompts, picks)) / len(picks):.3f}")

    # ---- 4. Rejection sampling: fine-tune on the selected answers ----
    tuned, kept = rejection_sampling(gen, rm, rs_prompts, n=8)
    control, kept_control = rejection_sampling(gen, rm, rs_prompts, n=1)
    kept_gold = sum(gold_score(p, c) for p, c in kept) / len(kept)
    control_gold = sum(gold_score(p, c) for p, c in kept_control) / len(kept_control)
    print("\n[4] rejection sampling (gold score on held-out prompts)")
    print(f"    kept data: best of 8 {kept_gold:.3f}, control with no selection (n = 1) {control_gold:.3f}")
    rows = [("instruction-tuned generator", gen), ("fine-tuned on best of 8", tuned), ("control, fine-tuned on n = 1", control)]
    for name, model in rows:
        sampled = mean_gold_score(model, eval_prompts)
        greedy = mean_gold_score(model, eval_prompts, n=1, temperature=0)
        print(f"    {name:<30} sampled {sampled:.3f}   greedy {greedy:.3f}")
    print("    greedy answers for 'abcd', 'hgfe', 'cafe':")
    for name, model in rows:
        print(f"    {name:<30}", [sample_completions(model, p, 1, temperature=0)[0] for p in ("abcd", "hgfe", "cafe")])


if __name__ == "__main__":
    main()

