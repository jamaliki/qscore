This is a reimplementation of Pintilie et al., 2020 [1] using pure Python. It is also faster since it does not use Chimera and tries to vectorize as many things as possible. Please check the implementation to make sure the results are correct.
To use this, you can run the `calculate_per_residue_q_scores` function in `q_score.py`

1 - Pintilie, G., Zhang, K., Su, Z. et al. Measurement of atom resolvability in cryo-EM maps with Q-scores. Nat Methods 17, 328–334 (2020). https://doi.org/10.1038/s41592-020-0731-1