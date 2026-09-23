## Hyperplane Dropout

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22910849.svg)](https://doi.org/10.5281/zenodo.22910849)

### Description
> We present an empirical study of geometry-aware adaptive dropout for fully-connected
networks. Each ReLU neuron is parameterized as a hyperplane (orientation, data-aware
position), and we derive three redundancy/importance signals: orientation (cosine
similarity of normal vectors), position embedding (mean activation / activation
consistency), and functional impact (activation x downstream weight). We implement a
training-time adaptive dropout layer (HyperplaneDropout) with EMA-smoothed probability
updates, evaluated on MNIST ablations and 2D decision-boundary stress tests. We introduce
Island Count, a topological metric quantifying isolated decision regions, and show the
method reduces boundary fragmentation. We also report negative results: (i) conjunctive
(AND) multi-criteria punishment under-punishes redundancy and underperforms
orientation-only punishment; (ii) the hypothesis that functional impact proxies geometric
position is refuted by coefficient-of-variation and Jaccard diagnostics; (iii) we discuss
how geometric redundancy criteria relate to, and are partly subsumed by,
information-theoretic (mutual-information) redundancy criteria. Code, figures, and
experimental logs are released for reproducibility.