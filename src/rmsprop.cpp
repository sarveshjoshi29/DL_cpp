#include "rmsprop.hpp"
#include "Layer.hpp"
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
double RMSProp::square(double x) {
	return x * x;
}

double RMSProp::rt(double x) {
	return sqrt(x);
}

RMSProp::RMSProp(double beta, double alpha) : alpha{alpha}, beta{beta} {
}

void RMSProp::init_params(std::vector<nn::Layer*>& layers) {
	sdw.resize(layers.size());
	sdb.resize(layers.size());
	// std::cout << "fine\n";
	for(int i = 0; i < sdw.size(); i++) {
		if(!layers[i]->is_trainable) {
			continue;
		}
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

void RMSProp::update_params(std::vector<nn::Layer*>& layers) {
	double epsilon = 1e-8;
	for(size_t i = 0; i < layers.size(); i++) {
		if(!layers[i]->is_trainable) {
			continue;
		}
		sdw[i] = sdw[i] * beta + layers[i]->dw.apply_elem_wise(square) * (1 - beta);
		layers[i]->w = layers[i]->w - (layers[i]->dw * alpha) / (sdw[i].apply_elem_wise(rt) + epsilon);

		sdb[i] = sdb[i] * beta + layers[i]->db.apply_elem_wise(square) * (1 - beta);
		layers[i]->b = layers[i]->b - (layers[i]->db * alpha) / (sdb[i].apply_elem_wise(rt) + epsilon);
	}
}

} // namespace nn