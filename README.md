# SCR2ST
SCR²-ST is a unified framework that leverages single-cell prior knowledge to guide efficient data acquisition and accurate expression prediction.

![Figure1_proposal](.\asset\Figure1_proposal.png)

**Comparison between traditional ST sampling and our active sampling.** 
*Left:* Traditional ST methods rely on fixed-grid sampling regardless of biological importance, leading to redundant measurements in similar regions and inefficient use of sequencing budgets. *Right:* Our proposed approach actively selects informative spots by incorporating single-cell prior knowledge, reducing redundancy while preserving biologically diverse regions.

## Overview

![Figure2_framework](.\asset\Figure2_framework.png)

This framework addresses the challenge of predicting gene expression from histology images in spatial transcriptomics. We propose a reinforcement learning-based active sampling strategy that intelligently selects informative spots for training by leveraging:

- **Single-cell manifold coverage**: Ensures sampled spots cover diverse cell states in the scRNA-seq reference
- **Cell type diversity**: Maximizes the entropy of cell type distribution in the selected samples
- **Spatial uniformity**: Encourages spatially dispersed sampling for better tissue coverage

The model also incorporates a retrieval-augmented module that enhances predictions by referencing similar expression patterns from the training set.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py \
    --dataset HER2 \
    --root_path /path/to/st_data \
    --patch_root /path/to/patches \
    --sc_root /path/to/sc_embeddings \
    --total_ratio 0.5 \
    --max_epochs 100 \
    --batch_size 128 \
    --gpu 0
```

## Project Structure

```
├── main.py           # Training script
├── model.py          # Model architectures
├── dataset.py        # Dataset class
├── rl_sampler.py     # RL-based sampler
├── reward.py         # Reward functions
├── cross_fold.py     # Cross-validation splits
├── eval_metric.py    # Evaluation metrics
└── requirements.txt  # Dependencies
```
