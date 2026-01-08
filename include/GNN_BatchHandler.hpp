#pragma once
#include "Graph.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "optimizer.hpp"
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
class GNN_BatchHandler {
	public:
		size_t batch_size;
		GNN_BatchHandler(std::vector<nn::Graph*> disjoint_graphs, std::vector<matx::Matrix<double>> batches_Y,
						 matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>), nn::optimizer* update_manager,
						 double (*loss)(matx::Matrix<double>, matx::Matrix<double>));

		matx::Matrix<double> out_last;
		double curr_loss;
		std::vector<nn::Graph*> graphs;
		std::vector<matx::Matrix<double>> batches_Y;
		std::vector<matx::Matrix<int>> adj_matrices;
		std::vector<matx::Matrix<double>> feature_vecs;

		matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>);
		double (*loss)(matx::Matrix<double>, matx::Matrix<double>);
		nn::optimizer* update_manager;

		void forward_prop_itr(size_t itr, std::vector<nn::Layer*>& layers);
		void backward_prop_itr(size_t itr, std::vector<nn::Layer*>& layers);

		void update_params(std::vector<nn::Layer*>& layers);
		void init_params_updmanager(std::vector<nn::Layer*>& layers);
		double compute_loss(matx::Matrix<double>& y);
		void one_epoch(std::vector<nn::Layer*>& layers, int verbose);
};
} // namespace nn