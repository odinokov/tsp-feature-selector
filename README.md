# TSP Feature Selector

**TSPFeatureSelector** implements the Top-Scoring Pairs (TSP) algorithm for feature selection in gene expression data, with efficient computation using Numba.

For more details on the TSP algorithm:
- [Top-scoring pairs for feature selection](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-12-375)
- [Applications to cancer outcome prediction](https://academic.oup.com/bioinformatics/article/21/20/3896/203010)
- [A Pairwise Feature Selection Method For Gene Data Using Information Gain](https://egrove.olemiss.edu/etd/943/)
- [A Robust and Efficient Feature Selection Algorithm for Microarray Data](https://onlinelibrary.wiley.com/doi/10.1002/minf.201600099)
- [The Robust Classification Model Based on Combinatorial Features](https://ieeexplore.ieee.org/document/8126830/)

## Status: Work in Progress

This implementation is under development and requires validation.

## Installation

Clone the repository and install with Poetry:

```bash
git clone https://github.com/odinokov/tsp-feature-selector.git
cd tsp-feature-selector
poetry install
```

## Usage

```python
from tspfs import TSPFeatureSelector
import numpy as np

# Simulate gene expression data
num_genes = 500
num_samples = 1000
np.random.seed(42)

gene_expression = np.random.rand(num_genes, num_samples)
class_labels = np.array([True] * 500 + [False] * 500)

assert np.all(np.isin(class_labels, [True, False])) # Class labels must contain only True and False values

# Simulate up/down-regulated genes for both classes
gene_expression[:2, class_labels] *= 2.0  # Upregulated genes for class 1
gene_expression[2:4, class_labels] *= 0.5  # Downregulated genes for class 1
gene_expression[10, class_labels] *= 0.5  # Another downregulated gene for class 1

gene_expression[:2, ~class_labels] *= 0.5  # Downregulated genes for class 0
gene_expression[2:4, ~class_labels] *= 2.0  # Upregulated genes for class 0
gene_expression[10, ~class_labels] *= 2.0  # Upregulated genes for class 0

# Select top 10 gene pairs
selector = TSPFeatureSelector(top_n=10)
selector.fit(gene_expression, class_labels)

# Output top pairs and delta scores
top_pairs = selector.get_top_pairs()
top_scores = selector.get_top_scores()

for i, (gene_i, gene_j) in enumerate(top_pairs):
    print(f"Pair {i+1}: Gene {gene_i}, Gene {gene_j}, Delta: {top_scores[i]:.2f}")
```

## License

MIT License.
