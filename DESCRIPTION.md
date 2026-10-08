# HOP-RAL — Project Description

## Title

**HOP-RAL: Replay-Free Analytic Learning for Class-Incremental Node Classification on Graphs**

## Short Description

HOP-RAL is a replay-free, backpropagation-free method for class-incremental node classification on graphs. It uses a fixed random graph encoder followed by a class-balanced ridge classifier whose sufficient statistics are updated incrementally and solved analytically.

## Problem

In class-incremental graph learning, new classes arrive sequentially while previous training data may no longer be available. Standard gradient-trained graph neural networks can suffer from catastrophic forgetting when adapting to each new task.

HOP-RAL investigates whether a graph node classifier can learn such a stream without storing training nodes and without performing gradient updates.

## Proposed Solution

HOP-RAL separates representation construction from classifier learning.

### 1. Fixed Graph Encoder

The method:

- normalizes input features,
- applies a seeded Gaussian random projection,
- performs K-hop graph propagation,
- concatenates propagation hops,
- applies a fixed random nonlinear expansion.

No encoder parameters are learned.

### 2. Recursive Analytic Classifier

For every task, class-balanced statistics are added to two matrices:

- `G`: Gram matrix
- `Q`: feature/label cross-product

The classifier is then solved using a ridge-regression objective:

`W = (G + λI)^(-1) Q`

The solution is obtained using Cholesky factorization.

### 3. Test-Time Score Smoothing

At inference, HOP-RAL performs one-hop score smoothing with `α = 0.5`. This does not modify the stored training statistics.

## Main Contribution

Under the CGLB class-incremental protocol, when there are no edges between tasks, HOP-RAL's classifier after each task is equivalent, up to floating-point rounding, to the class-balanced ridge classifier trained jointly on all training nodes seen so far.

This removes the need to replay previous examples while avoiding representation drift caused by continued gradient training.

## Default Configuration

- Random projection dimension: `m = 512`
- Propagation depth: `K = 2`
- Random expansion dimension: `De = 2048`
- Final feature width: `D = 3584`
- Score smoothing coefficient: `α = 0.5`
- Smoothing hops: `1`

## Datasets

The reported evaluation uses:

1. **CoraFull**
   - 19,793 nodes
   - 8,710 features
   - 70 classes
   - 35 tasks

2. **Amazon Computers**
   - 13,752 nodes
   - 767 features
   - 10 classes
   - 5 tasks

3. **Roman-empire**
   - 22,662 nodes
   - 300 features
   - 18 classes
   - 9 tasks
   - Edge homophily: 0.05

Each class uses a 60%/20%/20% train/validation/test split.

## Reported Performance

Across five seeds, HOP-RAL reports:

| Dataset | AP | BWT |
|---|---:|---:|
| CoraFull | 79.3 ± 0.7% | −3.2 ± 0.3 |
| Amazon Computers | 96.8 ± 0.2% | −1.3 ± 0.4 |
| Roman-empire | 62.2 ± 0.7% | −12.1 ± 0.7 |

The manuscript reports that regularization and replay baselines implemented with a three-layer GCN showed substantially lower final average performance in the same experimental setup.

## Memory

HOP-RAL does not retain training nodes, subgraphs, or condensed graphs. Its persistent state consists of `G` and `Q`.

Because the Gram matrix scales quadratically with feature width, the method trades sample storage for parameter-statistics storage. On CoraFull, the reported state requires approximately 104.8 MB in double precision.

## Computational Profile

The method does not require training epochs. The reported CoraFull stream, including evaluation of five candidate ridge strengths, took approximately 48 seconds on one CPU thread in the described experimental environment.

The per-task computational costs are:

- `O(D²)` per training node for statistics updates.
- `O(D³)` for solving the ridge system.

## Key Findings

- Propagation is especially important on the homophilous CoraFull and Computers graphs.
- Random nonlinear expansion improves performance, particularly on Roman-empire.
- Hop concatenation improves Roman-empire performance but slightly reduces performance on CoraFull and Computers.
- Score smoothing improves CoraFull performance more than the other datasets.
- Class balancing improves CoraFull and Roman-empire performance, with a dataset-dependent effect on backward transfer.
- HOP-RAL does not require a pretrained graph encoder or a labeled base session.

## Limitations

The current study is limited by:

- moderate-sized evaluated graphs,
- a fixed class order,
- absence of large-scale datasets such as ogbn-arxiv,
- the `O(D²)` persistent state,
- the assumption of no inter-task edges for the exact analytic equivalence,
- comparisons with CaT and PUMA that rely on published results from other implementations,
- ridge hyperparameter selection using validation information after the stream.

## Intended Repository Use

A code repository implementing HOP-RAL can contain:

- the fixed graph encoder,
- recursive class-balanced ridge learning,
- score smoothing,
- CGLB-style data preparation,
- baseline implementations,
- experiment scripts,
- configuration files,
- result aggregation and visualization.

The manuscript states that code was available from the authors on request.

## Authors

P. Preetha  
J. Adhava  
Catherine Seby  
V. Aakash

Department of Artificial Intelligence and Machine Learning  
KPR Institute of Engineering and Technology, Coimbatore, India
