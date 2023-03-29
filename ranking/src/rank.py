import numpy as np
import pandas as pd


class Topsis:
    decision_matrix = np.array([])  # Matrix
    combinations = []
    weighted_normalized_decision_matrix = np.array([])  # Weight matrix
    normalized_decision_matrix = np.array([])  # Normalisation matrix
    M = 0  # Number of rows
    N = 0  # Number of columns

    """
	Create an evaluation matrix consisting of m alternatives and n criteria,
	with the intersection of each alternative and criteria given as {\displaystyle x_{ij}}x_{ij},
	we therefore have a matrix {\displaystyle (x_{ij})_{m\times n}}(x_{{ij}})_{{m\times n}}.
	"""

    def __init__(self, decision_matrix: np.ndarray, weight_matrix: list[int], criteria: list[bool]):
        self.dmatrix = decision_matrix

        # M×N matrix
        self.decision_matrix = np.array(decision_matrix, dtype="float")

        # M alternatives (options)
        self.row_size = len(self.decision_matrix)

        # N attributes (criteria)
        self.column_size = len(self.decision_matrix[0])

        # N size weight matrix
        self.weight_matrix = np.array(weight_matrix, dtype="float")
        self.weight_matrix = self.weight_matrix / sum(self.weight_matrix)
        self.criteria = np.array(criteria, dtype="float")

    """
	# Step 2
	The matrix {\displaystyle (x_{ij})_{m\times n}}(x_{{ij}})_{{m\times n}} is then normalised to form the matrix
	"""

    def step_2(self):
        # normalized scores
        self.normalized_decision_matrix = self.decision_matrix.copy()

        sqrd_sum = np.power(self.normalized_decision_matrix, 2).sum(axis=0)

        self.normalized_decision_matrix = self.normalized_decision_matrix / (sqrd_sum**0.5)

    """
	# Step 3
	Calculate the weighted normalized decision matrix
	"""

    def step_3(self):
        self.weighted_normalized_decision_matrix = self.normalized_decision_matrix.copy()

        self.weighted_normalized_decision_matrix = self.weight_matrix * self.weighted_normalized_decision_matrix

    """
	# Step 4
	Determine the worst alternative {\displaystyle (A_{w})}(A_{w}) and the best alternative {\displaystyle (A_{b})}(A_{b}):
	"""

    def step_4(self):
        self.worst_alternatives = np.zeros(self.column_size)
        # self.best_alternatives = np.zeros(self.column_size)

        self.best_alternatives = np.array([0, 0, 0])  # set custom best alternatives

        for i in range(self.column_size):
            if self.criteria[i]:
                self.worst_alternatives[i] = min(self.weighted_normalized_decision_matrix[:, i])
                # self.best_alternatives[i] = max(self.weighted_normalized_decision_matrix[:, i])
            else:
                self.worst_alternatives[i] = max(self.weighted_normalized_decision_matrix[:, i])
                # self.best_alternatives[i] = min(self.weighted_normalized_decision_matrix[:, i])

    """
	# Step 5
	Calculate the L2-distance between the target alternative {\displaystyle i}i and the worst condition {\displaystyle A_{w}}A_{w}
	{\displaystyle d_{iw}={\sqrt {\sum _{j=1}^{n}(t_{ij}-t_{wj})^{2}}},\quad i=1,2,\ldots ,m,}
	and the distance between the alternative {\displaystyle i}i and the best condition {\displaystyle A_{b}}A_b
	{\displaystyle d_{ib}={\sqrt {\sum _{j=1}^{n}(t_{ij}-t_{bj})^{2}}},\quad i=1,2,\ldots ,m}
	where {\displaystyle d_{iw}}d_{{iw}} and {\displaystyle d_{ib}}d_{{ib}} are L2-norm distances
	from the target alternative {\displaystyle i}i to the worst and best conditions, respectively.
	"""

    def step_5(self):
        self.worst_distance = np.zeros(self.row_size)
        self.best_distance = np.zeros(self.row_size)

        self.worst_distance_mat = self.weighted_normalized_decision_matrix.copy()
        self.best_distance_mat = self.weighted_normalized_decision_matrix.copy()

        self.worst_distance = ((self.worst_distance_mat - self.worst_alternatives) ** 2).sum(axis=1) ** 0.5
        self.best_distance = ((self.best_distance_mat - self.best_alternatives) ** 2).sum(axis=1) ** 0.5

    """
	# Step 6
	Calculate the similarity
	"""

    def step_6(self):
        np.seterr(all="ignore")
        self.worst_similarity = np.zeros(self.row_size)
        self.best_similarity = np.zeros(self.row_size)
        self.performance_score = np.zeros(self.row_size)

        self.worst_similarity = self.worst_distance / (self.worst_distance + self.best_distance)
        self.best_distance = self.worst_distance / (self.worst_distance + self.best_distance)

    def rank_to_worst_similarity(self):
        return [ind + 1 for ind, val in enumerate(self.worst_similarity.argsort())]

    def rank_to_best_similarity(self):
        return [val + 1 for val in self.best_similarity.argsort()]

    def calc(self):
        # print("Step 1\n", self.decision_matrix, end="\n\n")
        self.step_2()
        # print("Step 2\n", self.normalized_decision_matrix, end="\n\n")
        self.step_3()
        # print("Step 3\n", self.weighted_normalized_decision_matrix, end="\n\n")
        self.step_4()
        # print("Step 4\n", self.worst_alternatives, self.best_alternatives, end="\n\n")
        self.step_5()
        # print("Step 5\n", self.worst_distance, self.best_distance, end="\n\n")
        self.step_6()
        # print("Step 6\n", self.worst_similarity, self.best_similarity, end="\n\n")

    def get_rank(self):
        self.calc()
        scores = pd.DataFrame(
            {
                "performance": self.worst_similarity,
            },
            index=self.dmatrix.index,
        )
        # Append additional previous details
        norm_dmatrix = pd.DataFrame(
            self.weighted_normalized_decision_matrix,
            columns=["norm_cancellables", "norm_space", "norm_distance"],
            index=self.dmatrix.index,
        )
        scores = pd.concat([scores, self.dmatrix, norm_dmatrix], axis=1)

        scores = scores.sort_values(by="performance", ascending=False)

        # Adding ranking column to score df
        scores.insert(0, "rank", range(1, len(self.worst_similarity) + 1))

        # Save the score results to csv file
        # scores.to_csv(settings.COMB_SCORES_PATH)

        return scores
