# HOP-RAL

## Replay-Free Analytic Learning for Class-Incremental Node Classification on Graphs

HOP-RAL is a replay-free and backpropagation-free approach to **class-incremental node classification on graphs**. It combines a fixed, training-free graph encoder with a recursive class-balanced ridge classifier.

The central idea is to learn a node classifier without retaining previous training nodes, subgraphs, or synthetic graphs. Instead, HOP-RAL maintains only two sufficient-statistics matrices and updates them task by task.

> **Paper:** HOP-RAL: Replay-Free Analytic Learning for Class-Incremental Node Classification on Graphs

### Authors

- P. Preetha
- J. Adhava
- Catherine Seby
- V. Aakash
- Department of Artificial Intelligence and Machine Learning
- KPR Institute of Engineering and Technology, Coimbatore, India

---

## Overview

Continual graph learning becomes challenging when new classes arrive over time because gradient-based graph neural networks can suffer from catastrophic forgetting.

HOP-RAL addresses this setting by removing both major sources of continual-learning overhead:

1. **No replay** — previous training nodes, subgraphs, and synthetic graphs are not stored.
2. **No backpropagation** — the encoder is fixed and the classifier is solved analytically.

For each task, HOP-RAL:

1. Normalizes the input node features.
2. Projects them using a fixed Gaussian random projection.
3. Performs K-hop graph propagation.
4. Optionally concatenates the propagated hops.
5. Applies fixed random nonlinear expansion.
6. Updates class-balanced ridge-regression statistics.
7. Solves the classifier in closed form.
8. Applies one-hop score smoothing at inference time.

Under the CGLB class-incremental protocol, when there are no edges between tasks, the resulting classifier is equivalent to the class-balanced ridge solution obtained from all training nodes seen so far.

---

## Architecture

HOP-RAL contains two main stages.

### Stage 1 — Training-Free Graph Encoder

The encoder has no trainable parameters.

**Input features → normalization → random projection → K-hop propagation → hop concatenation → random nonlinear expansion**

Default configuration:

| Parameter | Value |
|---|---:|
| Random projection dimension (`m`) | 512 |
| Propagation depth (`K`) | 2 |
| Random expansion dimension (`De`) | 2048 |
| Final feature width (`D`) | 3584 |
| Score smoothing coefficient (`α`) | 0.5 |
| Smoothing hops | 1 |

The random projection uses a Gaussian matrix with entries sampled from `N(0, 1/m)`. Graph propagation uses normalized adjacency with self-loops. Random nonlinear features are then generated using a fixed Gaussian matrix.

### Stage 2 — Recursive Class-Balanced Ridge Classifier

For each task, the method computes class-balanced sufficient statistics and adds them to the running state:

- `G` — Gram matrix
- `Q` — feature/label cross-product matrix

The classifier is obtained by solving:

`W = (G + λI)^(-1) Q`

using Cholesky factorization in double precision.

Only `G` and `Q` are retained between tasks. No previous training examples are stored.

---

## Why HOP-RAL?

| Property | HOP-RAL |
|---|---|
| Gradient updates | No |
| Replay of training nodes | No |
| Synthetic graph replay | No |
| Fixed pretrained backbone required | No |
| Base session required | No |
| Training epochs | None |
| Persistent state | `G` and `Q` |
| Class balancing | Yes |
| Graph propagation | Yes |
| Random nonlinear expansion | Yes |
| Closed-form classifier | Yes |

A key advantage is that the first two-class task is handled in exactly the same way as later tasks; there is no special base-training phase.

---

## Experimental Setup

Experiments were performed under the CGLB class-incremental protocol using:

### CoraFull

- 19,793 nodes
- 8,710 bag-of-words features
- 70 classes
- 35 two-class tasks

### Amazon Computers

- 13,752 nodes
- 767 bag-of-words features
- 10 classes
- 5 two-class tasks

### Roman-empire

- 22,662 nodes
- 300 word-embedding features
- 18 classes
- 9 two-class tasks
- Edge homophily: 0.05

Within each class, the split is:

- 60% training
- 20% validation
- 20% testing

The experiments use five seeds and report mean ± standard deviation.

---

## Results

### Final Class-Incremental Performance

| Method | CoraFull AP | Computers AP | Roman-empire AP |
|---|---:|---:|---:|
| Fine-tuning | 2.7 ± 0.2 | 19.8 ± 0.1 | 9.7 ± 0.0 |
| EWC | 2.5 ± 0.4 | 19.7 ± 0.0 | 9.7 ± 0.1 |
| LwF | 2.6 ± 0.3 | 19.7 ± 0.1 | 9.7 ± 0.1 |
| ER-GNN | 3.0 ± 0.4 | 20.0 ± 0.4 | 19.9 ± 1.6 |
| AL-GNN-style, 1-task base | 9.7 ± 0.6 | 56.7 ± 0.5 | 48.3 ± 1.7 |
| AL-GNN-style, half base | 57.7 ± 1.6 | 79.4 ± 1.4 | 55.3 ± 0.8 |
| **HOP-RAL** | **79.3 ± 0.7** | **96.8 ± 0.2** | **62.2 ± 0.7** |
| Joint GCN | 75.4 ± 1.4 | 95.7 ± 0.3 | 59.1 ± 1.0 |

HOP-RAL achieves:

- **79.3 ± 0.7% AP** on CoraFull
- **96.8 ± 0.2% AP** on Amazon Computers
- **62.2 ± 0.7% AP** on Roman-empire

Its BWT values are:

- CoraFull: **−3.2 ± 0.3**
- Computers: **−1.3 ± 0.4**
- Roman-empire: **−12.1 ± 0.7**

The manuscript notes that comparisons with published CaT and PUMA results are indicative rather than controlled because those results come from other implementations.

---

## Ablation Findings

The ablation study evaluates the contribution of individual components.

| Variant | CoraFull AP | Computers AP | Roman-empire AP |
|---|---:|---:|---:|
| HOP-RAL, K=2 | 79.3 ± 0.7 | 96.8 ± 0.2 | 62.2 ± 0.7 |
| Without propagation | 63.5 ± 1.5 | 88.5 ± 0.7 | 59.6 ± 0.6 |
| Without hop concatenation | 80.3 ± 0.5 | 97.1 ± 0.1 | 58.2 ± 0.8 |
| Without random expansion | 76.9 ± 0.9 | 96.5 ± 0.1 | 58.4 ± 1.0 |
| Without class balancing | 77.8 ± 0.8 | 96.8 ± 0.1 | 60.9 ± 1.2 |
| Without score smoothing | 76.8 ± 0.8 | 96.6 ± 0.3 | 61.7 ± 0.7 |

Propagation provides the largest improvement on the homophilous datasets. Hop concatenation behaves differently across graph types: it helps Roman-empire but slightly reduces performance on CoraFull and Computers.

---

## Computational Cost

HOP-RAL requires no training epochs.

The per-task update costs:

- `O(D²)` per training node for updating the Gram statistics.
- `O(D³)` for solving the ridge system.

The persistent state is:

- `G ∈ R^(D×D)`
- `Q ∈ R^(D×C)`

Thus, the state grows as `O(D²)`.

For CoraFull, the manuscript reports approximately:

- **104.8 MB** in double precision for the state.
- **52.4 MB** when stored in single precision.
- Storing only the upper triangle of the symmetric `G` can approximately halve the storage.

A complete CoraFull stream, including the five candidate ridge strengths, took approximately **48 seconds on one CPU thread** in the reported setup.

---

## Reproducibility

Reported implementation configuration:

- PyTorch 2.14
- One thread of a two-core Intel Xeon CPU at 2.1 GHz
- Two runs in parallel
- Five experimental seeds: 0–4
- Fixed class order
- `m = 512`
- `K = 2`
- `De = 2048`
- `α = 0.5`
- One smoothing hop
- Ridge strength selected from:
  - `10^-4`
  - `10^-3`
  - `10^-2`
  - `10^-1`
  - `1`

The manuscript states that code was available from the authors on request.

---

## Important Assumption

The exact joint-learning equivalence described by the paper depends on the CGLB-style setting where **no edges connect different tasks**.

If inter-task edges are introduced, propagated features of earlier nodes can change when new nodes arrive. In that setting, the stored statistics can become stale and the stated equivalence no longer holds.

---

## Limitations

The manuscript identifies several limitations:

- CaT and PUMA comparisons use published results from other implementations.
- The joint GCN on CoraFull is below the published joint-training result.
- EWC and LwF strengths were not tuned in the reported experiments.
- The class order is fixed.
- The evaluated graphs are moderate in size.
- Large graphs such as ogbn-arxiv were not tested.
- The only heterophilous dataset is also sparse after inter-task edges are removed.
- The `O(D²)` state can become expensive for larger feature widths.
- The reported ridge strength is selected using validation nodes after the stream, which is not strictly online.

---

## Project Structure

A suggested repository structure is:

```text
HOP-RAL/
├── README.md
├── DESCRIPTION.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   └── README.md
├── hop_ral/
│   ├── encoder.py
│   ├── ridge.py
│   ├── smoothing.py
│   ├── data.py
│   └── evaluation.py
├── baselines/
│   ├── finetuning.py
│   ├── ewc.py
│   ├── lwf.py
│   ├── er_gnn.py
│   └── al_gnn_style.py
├── experiments/
│   ├── run_corafull.py
│   ├── run_computers.py
│   └── run_roman_empire.py
└── results/
    └── README.md
```

This structure is a repository organization suggestion; the manuscript itself does not provide this exact source tree.

---

## Citation

If you use HOP-RAL in your work, cite the associated manuscript:

```bibtex
@article{hopral,
  title   = {HOP-RAL: Replay-Free Analytic Learning for Class-Incremental Node Classification on Graphs},
  author  = {Preetha, P. and Adhava, J. and Seby, Catherine and Aakash, V.},
  journal = {Manuscript},
  year    = {2026}
}
```

---

## References

The method builds on prior work in continual learning, continual graph learning, analytic learning, graph convolution, graph propagation, random features, and graph benchmarks. The complete reference list is provided in the associated manuscript.
