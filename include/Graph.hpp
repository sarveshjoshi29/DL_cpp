#pragma once
#include "Matrix.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nn {
class Graph {
	public:
		matx::Matrix<int> adj_matrix;
		matx::Matrix<double> feature_vectors;
		matx::Matrix<double> A_hat;
		// combine operator
		Graph operator%(const Graph& other) const;

		Graph(matx::Matrix<int> adj, matx::Matrix<double> features);
		void compute_A_hat();
};
} // namespace nn
