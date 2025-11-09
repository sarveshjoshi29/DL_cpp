#pragma once
#include "Batch_Handler.hpp"
#include "MLP_Layer.hpp"
#include "Matrix.hpp"
#include "activations.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iomanip>
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
class wrapper {
	private:
		nn::BatchHandler* batch_handler;

	public:
		std::vector<nn::MLP_Layer*> layers;
		std::mt19937 gen;
		matx::Matrix<double> data;
		matx::Matrix<double> y;
		double (*loss)(matx::Matrix<double>, matx::Matrix<double>);
		matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>);
		nn::optimizer* update_manager;

		~wrapper();

		wrapper(std::initializer_list<nn::MLP_Layer*> inp_layers, nn::optimizer* optimizer, string loss = "mse");

		void train(const matx::Matrix<double> X_train, const matx::Matrix<double> y_train, int epochs = 100, size_t batch_size = 32);

		matx::Matrix<double> predict(matx::Matrix<double> X_test);

		void printwts();

	private:
		void init_batch_handler(size_t batch_size, const matx::Matrix<double>& data, const matx::Matrix<double>& y,
								matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>), optimizer* update_manager);
};
} // namespace nn
