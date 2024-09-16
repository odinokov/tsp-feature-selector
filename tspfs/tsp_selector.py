from typing import List, Tuple

import numpy as np
from numba import njit, prange
from scipy.stats import rankdata


class TSPFeatureSelector:
    """
    The TSPFeatureSelector class implements the top-scoring pairs (TSP) algorithm
    for feature selection in gene expression data.

    For more details on the TSP algorithm:
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-12-375
    https://academic.oup.com/bioinformatics/article/21/20/3896/203010
    """

    def __init__(self, top_n: int = 1):
        """
        Initializes the TSP feature selector.
        """
        self.top_n = top_n
        self.rank_matrix = None
        self.selected_tsp_pairs = []
        self.selected_delta_scores = []

    def validate_input_data(
        self, gene_expression: np.ndarray, class_labels: np.ndarray
    ) -> None:
        """
        Validates the gene expression matrix and class labels.
        """
        if gene_expression.ndim != 2:
            raise ValueError("Gene expression matrix must be 2-dimensional.")
        if gene_expression.shape[1] != class_labels.size:
            raise ValueError(
                "Number of samples in gene expression matrix must match length of class labels."
            )
        unique_classes = np.unique(class_labels)
        if unique_classes.size != 2:
            raise ValueError("Class labels must contain exactly two distinct values.")
        if np.isnan(gene_expression).any():
            raise ValueError(
                "Gene expression matrix contains NaN values. Please clean your data."
            )

    def generate_rank_matrix(self, gene_expression: np.ndarray) -> np.ndarray:
        """
        Converts gene expression matrix into a rank matrix.
        """
        return np.apply_along_axis(rankdata, axis=0, arr=gene_expression)

    def fit(self, gene_expression: np.ndarray, class_labels: np.ndarray) -> None:
        """
        Fits the TSP feature selector by identifying the top gene pairs.
        """
        self.validate_input_data(gene_expression, class_labels)
        self.rank_matrix = self.generate_rank_matrix(gene_expression)

        class1_indices = np.where(class_labels)[0]
        class2_indices = np.where(~class_labels)[0]

        delta_scores, gene_i_indices, gene_j_indices = compute_delta_scores_numba(
            self.rank_matrix, class1_indices, class2_indices
        )

        # Sort the gene pairs based on delta scores
        sorted_delta_scores, sorted_gene_i, sorted_gene_j = sort_gene_pairs_numba(
            delta_scores, gene_i_indices, gene_j_indices
        )

        # Select top N gene pairs
        selected_gene_i, selected_gene_j = select_top_tsp_numba(
            sorted_delta_scores,
            sorted_gene_i,
            sorted_gene_j,
            self.rank_matrix,
            class1_indices,
            class2_indices,
            top_n=self.top_n,
        )

        # Store the selected top pairs and their corresponding delta scores
        self.selected_tsp_pairs = [
            (selected_gene_i[i], selected_gene_j[i]) for i in range(self.top_n)
        ]
        self.selected_delta_scores = [sorted_delta_scores[i] for i in range(self.top_n)]

    def get_top_pairs(self) -> List[Tuple[int, int]]:
        """
        Returns the top gene pairs selected by the feature selector.
        """
        return self.selected_tsp_pairs

    def get_top_scores(self) -> List[float]:
        """
        Returns the delta scores of the top selected gene pairs.
        """
        return self.selected_delta_scores


# Helper methods optimized with Numba
@njit(parallel=True)
def compute_delta_scores_numba(
    rank_matrix: np.ndarray, class1_indices: np.ndarray, class2_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_genes = rank_matrix.shape[0]
    num_pairs = (num_genes * (num_genes - 1)) // 2
    delta_scores = np.zeros(num_pairs, dtype=np.float64)
    gene_i_indices = np.zeros(num_pairs, dtype=np.int32)
    gene_j_indices = np.zeros(num_pairs, dtype=np.int32)

    class1_size = class1_indices.size
    class2_size = class2_indices.size

    for i in prange(num_genes - 1):
        for j in range(i + 1, num_genes):
            prob_class1 = (
                np.sum(rank_matrix[i, class1_indices] < rank_matrix[j, class1_indices])
                / class1_size
            )
            prob_class2 = (
                np.sum(rank_matrix[i, class2_indices] < rank_matrix[j, class2_indices])
                / class2_size
            )
            delta = np.abs(prob_class1 - prob_class2)
            pair_idx = i * num_genes - (i * (i + 1)) // 2 + (j - i - 1)
            delta_scores[pair_idx] = delta
            gene_i_indices[pair_idx] = i
            gene_j_indices[pair_idx] = j

    return delta_scores, gene_i_indices, gene_j_indices


@njit(parallel=True)
def sort_gene_pairs_numba(
    delta_scores: np.ndarray, gene_i_indices: np.ndarray, gene_j_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_indices = np.argsort(-delta_scores)
    return (
        delta_scores[sorted_indices],
        gene_i_indices[sorted_indices],
        gene_j_indices[sorted_indices],
    )


@njit
def compute_rank_score_numba(
    rank_matrix: np.ndarray, gene_i: int, gene_j: int, class_indices: np.ndarray
) -> float:
    total_diff = 0.0
    num_samples = class_indices.size
    for idx in range(num_samples):
        sample = class_indices[idx]
        total_diff += rank_matrix[gene_i, sample] - rank_matrix[gene_j, sample]
    return total_diff / num_samples

@njit
def select_top_tsp_numba(
    sorted_delta_scores: np.ndarray,
    sorted_gene_i: np.ndarray,
    sorted_gene_j: np.ndarray,
    rank_matrix: np.ndarray,
    class1_indices: np.ndarray,
    class2_indices: np.ndarray,
    top_n: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:

    total_pairs = sorted_delta_scores.size
    selected_gene_i = np.empty(top_n, dtype=np.int32)
    selected_gene_j = np.empty(top_n, dtype=np.int32)

    # Compute rank differences for all gene pairs
    rank_diffs = np.empty(total_pairs, dtype=np.float64)
    for idx in prange(total_pairs):
        gene_i = sorted_gene_i[idx]
        gene_j = sorted_gene_j[idx]
        gamma_c1 = compute_rank_score_numba(rank_matrix, gene_i, gene_j, class1_indices)
        gamma_c2 = compute_rank_score_numba(rank_matrix, gene_i, gene_j, class2_indices)
        rank_diffs[idx] = np.abs(gamma_c1 - gamma_c2)

    # Identify the start indices of groups with the same delta score
    group_starts = np.empty(total_pairs, dtype=np.int32)
    group_count = 0
    group_starts[0] = 0

    for idx in range(1, total_pairs):
        if sorted_delta_scores[idx] != sorted_delta_scores[idx - 1]:
            group_count += 1
            group_starts[group_count] = idx
    total_groups = group_count + 1

    selected_count = 0

    # Process each group until top_n gene pairs are selected
    for group_idx in range(total_groups):
        start_idx = group_starts[group_idx]
        if group_idx < total_groups - 1:
            end_idx = group_starts[group_idx + 1]
        else:
            end_idx = total_pairs

        group_size = end_idx - start_idx
        group_rank_diffs = rank_diffs[start_idx:end_idx]

        # Sort indices within the group based on rank differences
        sorted_indices_within_group = np.argsort(-group_rank_diffs)

        for s in range(group_size):
            if selected_count < top_n:
                index = start_idx + sorted_indices_within_group[s]
                selected_gene_i[selected_count] = sorted_gene_i[index]
                selected_gene_j[selected_count] = sorted_gene_j[index]
                selected_count += 1
            else:
                break

        if selected_count >= top_n:
            break

    return selected_gene_i, selected_gene_j
