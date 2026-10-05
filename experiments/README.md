# Experiments

Follow-up experiments on top of the main GA runs in [`results/ga_runs`](../results/ga_runs).
Each folder holds the script that produced the data, the raw output and a short text summary.

The scripts call a paid API. You do not need to run them to check the numbers: the committed
outputs are enough, and `python analysis/make_figures.py` rebuilds the bankruptcy and
cost-sensitivity tables from them.

| Folder | Question | Who plays what | Calls to re-run |
|---|---|---|---|
| [`bankruptcy`](bankruptcy) | How often does a firm go bankrupt (3 losing periods in a row)? 20 markets, each with 1 optimized and 9 random-prompt firms. | `gpt-4o-mini` plays genome `11100110` | 2,000 |
| [`cost_sensitivity`](cost_sensitivity) | Does a genome evolved at cost index k = 0.75 hold up at k = 0.3, 0.5 and 0.9? 10 markets per k. | `gpt-4o-mini` plays genome `10011100` | 4,000 |
| [`reproducibility`](reproducibility) | How repeatable are the model at temperature 0 and the GA? | `gpt-4o-mini` | 200 (variance test), 22,500 (GA re-run) |

Calls = markets x firms x periods (x generations for the GA).

## Reading the results

**Bankruptcy.** The optimized firm goes bankrupt in 0 of 20 markets, against 52 of 180 random-prompt firms
(28.9%). Mean profits are close (6,453 vs. 4,073) with a very wide spread for the random firms, so this
experiment says more about survival than about profit. The genome `11100110` was evolved for
`gpt-3.5-turbo`, the model of the 20-market comparison in
[`results/optimized_vs_baseline`](../results/optimized_vs_baseline), but the committed run was played by
`gpt-4o-mini` (the default `LLM_MODEL`), so it is a transfer setting. For a matched run:
`LLM_MODEL=gpt-3.5-turbo python experiments/bankruptcy/run_bankruptcy.py`.

**Cost sensitivity.** The genome evolved at k = 0.75 is specialised to that regime. At k = 0.3 and 0.5
it never goes bankrupt but earns less than the random-prompt firms on average (8.7k vs. 28.0k and
2.4k vs. 9.3k). At k = 0.75 it earns 22.1k vs. 2.3k. At k = 0.9 it goes bankrupt in 5 of 10 markets,
and so do many random firms (37 of 90).

**Reproducibility.**
- *Variance test.* The same optimized prompt was sent 100 times in each of two scenarios and 94-97% of
  the replies were identical at temperature 0, with two distinct decisions in total. The two scenarios
  send the same messages (the prompt carries no market state), so this is 200 calls of one prompt.
- *GA re-run.* Three more GA runs, with the same settings, gave the genomes `01010101`, `10010100` and
  `10010100`; the original three were `01010100`, `10011100` and `10011100`. The most frequent genome,
  `10011100`, appears in 2 of the 6 runs; the re-runs are 1 to 4 bits away from the originals.
  Hosted models are not fully deterministic, so a GA run is repeatable up to a few bits, not exactly.

[`reproducibility/analyze_reproducibility.py`](reproducibility/analyze_reproducibility.py) runs offline
on the committed logs.
