#pragma once
#include "Layer.hpp"
#include "MLP_Layer.hpp"
#include "Matrix.hpp"
#include "loss.hpp"
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
class BatchHandler {
	public:
		size_t batch_size;
		matx::Matrix<double> out_last;
		double curr_loss;
		std::vector<matx::Matrix<double>> batches_X;
		std::vector<matx::Matrix<double>> batches_Y;
		size_t num_matx;
		matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>);
		double (*loss)(matx::Matrix<double>, matx::Matrix<double>);
		nn::optimizer* update_manager;

		BatchHandler(size_t batch_size, const matx::Matrix<double>& data, const matx::Matrix<double>& y,
					 matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>), nn::optimizer* update_manager,
					 double (*loss)(matx::Matrix<double>, matx::Matrix<double>));

		void split_data(const matx::Matrix<double>& data, const matx::Matrix<double>& y);

		void forward_prop_itr(size_t itr, std::vector<nn::Layer*>& layers);

		void backward_prop_itr(size_t itr, std::vector<nn::Layer*>& layers);
		void update_params(std::vector<nn::Layer*>& layers);

		void init_params_updmanager(std::vector<nn::Layer*>& layers);

		void one_epoch(std::vector<nn::Layer*>& layers, int verbose);
		double compute_loss(matx::Matrix<double>& y);
};
} // namespace nn