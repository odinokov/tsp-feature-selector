import numpy as np

from tspfs import TSPFeatureSelector


def main():
    num_genes = 500
    num_samples = 1000
    np.random.seed(42)

    gene_expression = np.random.rand(num_genes, num_samples)
    class_labels = np.array([True] * 500 + [False] * 500)

    gene_expression[:2, class_labels] += 1
    gene_expression[2:4, class_labels] -= 1
    gene_expression[10, class_labels] -= 1

    gene_expression[:2, ~class_labels] -= 1
    gene_expression[2:4, ~class_labels] += 1
    gene_expression[10, ~class_labels] += 1

    print(
        f"Running test for {num_genes} genes and {num_samples} samples. Please wait..."
    )

    TOP_N = 10
    selector = TSPFeatureSelector(top_n=TOP_N)
    selector.fit(gene_expression, class_labels)

    top_pairs = selector.get_top_pairs()
    top_scores = selector.get_top_scores()

    print(f"Top {TOP_N} gene pairs:")
    for i, (gene_i, gene_j) in enumerate(top_pairs):
        delta_score = top_scores[i]
        print(
            f"Pair {i+1}: Gene {gene_i} and Gene {gene_j} with Delta Score: {delta_score:.2f}"
        )


if __name__ == "__main__":
    main()
