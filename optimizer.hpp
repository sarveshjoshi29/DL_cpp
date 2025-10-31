#pragma once
#include "MLP_Layer.hpp"
#include "Matrix.hpp"
#include "loss.hpp"
#include "wrapper.hpp"
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

// IDEA IS -- BATCH HANDLER WILL TAKE CARE OF BATCHES... FOR RMSPROP/MOMENTUM/ADAM I NEED TO CREATE CLASSES TO CHANGE MY UPDATE_PARAMS FUNCTION! SO I
// THINK I SHOULD PLAN TO REMOVE ONLY UPDATE_PARAMS FROM BATCH HANDLER BUT KEEP THE REST.
//
//
// DEFINE UPDATE_PARAMS(LAYERS,LEARNING_RATE) IN DIFF CLASSES AND PASS POINTER THROUGH CONSTRUCTOR OF BATCH HANDLER.

// need to debug batch handler --- done
// and all init_param , update_param providers :) --- ADAM rem.

// Structure -- templated wrapper and batch handler to handle diff optims. You can make a custom optimizer which MUST have two functions -
// 1) init_params(layers) -- this should initialized all the internal params required (can use size of each layer).
// 2) update_params(layers) -- this should update w,b of EACH Layer when dw,db are known and using internal parameters

// any other functions for custom optim will be used within that class only!

namespace nn {
class optimizer {
	public:
		template<typename O> class BatchHandler {
			public:
				size_t batch_size;
				std::vector<matx::Matrix<double>> batches_X;
				std::vector<matx::Matrix<double>> batches_Y;
				size_t num_matx;
				matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>);
				O* update_manager;

				BatchHandler(size_t batch_size, const matx::Matrix<double>& data, const matx::Matrix<double>& y,
							 matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>), O* update_manager)
					: batch_size{batch_size}, lossprime{lossprime}, update_manager{update_manager} {
					split_data(data, y);
				}

				void split_data(const matx::Matrix<double>& data, const matx::Matrix<double>& y) {

					size_t rows = data.size();
					size_t cols = data[0].size();
					size_t y_cols = y[0].size();
					num_matx = rows / batch_size;
					num_matx = (num_matx == 0 ? 1 : num_matx);
					// std::cout << "cols is " << cols << "\n";

					for(size_t i = 0; i < num_matx; i++) {
						matx::Matrix<double> temp = data(i * batch_size, min((i + 1) * batch_size, rows), 0, cols);
						// std::cout << "batch handler --- shape is --- " << temp.shape()[0] << " " << temp.shape()[1] << " ----\n";
						batches_X.push_back(temp);
						matx::Matrix<double> tempy = y(i * batch_size, min((i + 1) * batch_size, rows), 0, y_cols);
						batches_Y.push_back(tempy);
					}
				}

				void forward_prop_itr(size_t itr, std::vector<nn::MLP_Layer*>& layers) {
					//	std::cout << "forwin\n";
					for(size_t i = 0; i < layers.size(); i++) {
						if(i == 0) {
							(*layers[i]).forward_pass(batches_X[itr].transpose());
						} else {
							// std::cout << "0th layer --- " << layers[i - 1]->a.shape()[0] << " " << layers[i - 1]->a.shape()[1] << "\n";
							(*layers[i]).forward_pass(layers[i - 1]->a);
						}
					}
					//	std::cout << "forwout\n";
				}

				void backward_prop_itr(size_t itr, std::vector<nn::MLP_Layer*>& layers) {
					nn::MLP_Layer* last = layers[layers.size() - 1];
					//	std::cout << "backwin\n";
					matx::Matrix<double> delta = lossprime(batches_Y[itr].transpose(), last->a) * last->activationprime(last->z);

					for(int i = layers.size() - 1; i >= 0; i--) {
						matx::Matrix<double> a_prev = (i == 0) ? batches_X[itr].transpose() : layers[i - 1]->a;
						layers[i]->backward_pass(a_prev, delta);
						if(i) {
							delta = (layers[i]->w.transpose() % delta) * layers[i - 1]->activationprime(layers[i - 1]->z);
						}
					}
					//	std::cout << "backout\n";
				}

				// independent of itr , keep memory of required variables in the other provider class itself!
				void update_params(std::vector<nn::MLP_Layer*>& layers) {
					// std::cout << "update in\n";
					update_manager->update_params(layers);
				}

				void init_params_updmanager(std::vector<nn::MLP_Layer*>& layers) {

					update_manager->init_params(layers);
					//	std::cout << "exited ";
				}

				void one_epoch(std::vector<nn::MLP_Layer*>& layers) {
					// std::cout << num_matx << "\n";
					for(size_t itr = 0; itr < num_matx; itr++) {

						// std::cout << " itr is " << itr << "\n";
						forward_prop_itr(itr, layers);
						backward_prop_itr(itr, layers);
						update_params(layers);
					}
				}
		};

		class GD_Momentum {
			public:
				std::vector<matx::Matrix<double>> vdw;
				std::vector<matx::Matrix<double>> vdb;
				double beta;
				double alpha;

				GD_Momentum(double beta = 0.9, double alpha = 0.01) : beta{beta}, alpha{alpha} {
				}

				void init_params(std::vector<nn::MLP_Layer*>& layers) {
					vdw.resize(layers.size());
					vdb.resize(layers.size());
					// std::cout << "fine\n";
					for(int i = 0; i < vdw.size(); i++) {
						//	std::cout << layers[0]->dw.size() << "\n";
						std::vector<std::vector<double>> tempvec(layers[i]->dw.size(), std::vector<double>(layers[i]->dw[0].size(), 0));
						// std::cout << "okk\n";
						matx::Matrix<double> temp(tempvec);
						vdw[i] = temp;

						std::vector<std::vector<double>> tempvec2(layers[i]->db.size(), std::vector<double>(layers[i]->db[0].size(), 0));
						matx::Matrix<double> temp2(tempvec2);
						vdb[i] = temp2;
					}
				}

				void update_params(std::vector<nn::MLP_Layer*>& layers) {

					for(size_t i = 0; i < layers.size(); i++) {
						vdw[i] = vdw[i] * beta + (layers[i]->dw) * (1 - beta);
						layers[i]->w = layers[i]->w - (vdw[i] * alpha);

						vdb[i] = vdb[i] * beta + (layers[i]->db) * (1 - beta);
						layers[i]->b = layers[i]->b - (vdb[i] * alpha);
					}
				}
		};

		class RMSProp {
			public:
				static double square(double x) {
					return x * x;
				}

				static double rt(double x) {
					return sqrt(x);
				}

			public:
				std::vector<matx::Matrix<double>> sdw;
				std::vector<matx::Matrix<double>> sdb;
				double alpha;
				double beta;

				RMSProp(double beta = 0.5, double alpha = 0.01) {
				}

				void init_params(std::vector<nn::MLP_Layer*>& layers) {
					sdw.resize(layers.size());
					sdb.resize(layers.size());
					// std::cout << "fine\n";
					for(int i = 0; i < sdw.size(); i++) {
						//	std::cout << layers[0]->dw.size() << "\n";
						std::vector<std::vector<double>> tempvec(layers[i]->dw.size(), std::vector<double>(layers[i]->dw[0].size(), 0));
						// std::cout << "okk\n";
						matx::Matrix<double> temp(tempvec);
						sdw[i] = temp;

						std::vector<std::vector<double>> tempvec2(layers[i]->db.size(), std::vector<double>(layers[i]->db[0].size(), 0));
						matx::Matrix<double> temp2(tempvec2);
						sdb[i] = temp2;
					}
				}

				void update_params(std::vector<nn::MLP_Layer*>& layers) {

					for(size_t i = 0; i < layers.size(); i++) {

						sdw[i] = sdw[i] * beta + layers[i]->dw.apply_elem_wise(square);
						layers[i]->w = layers[i]->w - (layers[i]->dw * alpha) / (sdw[i].apply_elem_wise(rt));

						sdb[i] = sdb[i] * beta + layers[i]->db.apply_elem_wise(square);
						layers[i]->b = layers[i]->b - (layers[i]->db * alpha) / (sdb[i].apply_elem_wise(rt));
					}
				}
		};

		class Adam {
				// :)
		};
};
} // namespace nn
