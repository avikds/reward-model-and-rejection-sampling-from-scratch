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
