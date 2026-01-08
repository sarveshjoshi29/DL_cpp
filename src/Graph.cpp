#include "Graph.hpp"
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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nn {

Graph::Graph(matx::Matrix<int> adj, matx::Matrix<double> features) : adj_matrix{adj}, feature_vectors{features} {
}

Graph Graph::operator%(const Graph& other) const {
	size_t fin_size = adj_matrix.size() + other.adj_matrix.size();
	matx::Matrix<int> adj_final(fin_size, fin_size, 0);
	matx::Matrix<double> feature_final(fin_size, feature_vectors[0].size(), 0);

	if(feature_vectors[0].size() != other.feature_vectors[0].size()) {
		throw std::invalid_argument("Shapes of feature vector not compatible");
	}

	for(size_t i = 0; i < fin_size; i++) {
		for(size_t j = 0; j < fin_size; j++) {
			if(i < adj_matrix.size() && j < adj_matrix.size()) {
				adj_final[i][j] = adj_matrix[i][j];
			} else if(i < adj_matrix.size() && j >= adj_matrix.size()) {
				adj_final[i][j] = 0;
			}
			if(i >= adj_matrix.size() && j < adj_matrix.size()) {
				adj_final[i][j] = 0;
			}
			if(i >= adj_matrix.size() && j >= adj_matrix.size()) {
				adj_final[i][j] = other.adj_matrix[i][j];
			}
		}
	}

	for(size_t i = 0; i < fin_size; i++) {
		for(size_t j = 0; j < feature_vectors[0].size(); j++) {
			if(i < adj_matrix.size()) {
				feature_final[i][j] = feature_vectors[i][j];
			} else {
				feature_final[i][j] = other.feature_vectors[i - adj_matrix.size()][j];
			}
		}
	}

	Graph temp(adj_final, feature_final);
	return temp;
}

void Graph::compute_A_hat() {
	matx::Matrix<double> D(adj_matrix.size(), adj_matrix[0].size(), 0);
	for(int i = 0; i < adj_matrix.size(); i++) {
		for(int j = 0; j < adj_matrix[0].size(); j++) {
			if(adj_matrix[i][j]) {
				D[i][i]++;
			}
		}
	}

	matx::Matrix<double> D_inv(adj_matrix.size(), adj_matrix[0].size(), 0);
	for(int i = 0; i < adj_matrix.size(); i++) {
		if(D[i][i] != 0) {
			D_inv[i][i] = 1 / D[i][i];
		}
	}
	matx::Matrix<double> adj_double(adj_matrix.size(), adj_matrix[0].size(), 0);
	for(int i = 0; i < adj_matrix.size(); i++) {
		for(int j = 0; j < adj_matrix.size(); j++) {
			if(adj_matrix[i][j]) {
				adj_double[i][j] = 1;
			}
		}
	}
	A_hat = D_inv % adj_double;
}
} // namespace nn