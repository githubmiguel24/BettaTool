"""Perceptual Tier: HRNet-based probabilistic keypoint detection.

Outputs a mean vector + 2x2 covariance matrix per landmark rather than a
single fixed coordinate, per the thesis's Heatmap-Based Probabilistic
Keypoint Localization theory (Chapter 1).
"""
