#include "Adam.hpp"
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

// variables are mt , vt, alpha , beta1, beta2
double Adam::rt(double x) {
	return sqrt(x);
}

double Adam::square(double x) {
	return x * x;
}

Adam::Adam(double alpha, double beta1, double beta2) : alpha{alpha}, beta1{beta1}, beta2{beta2} {
	time = 1;
}

void Adam::init_params(std::vector<nn::Layer*>& layers) {

	m.resize(layers.size());
	v.resize(layers.size());
	m_bias.resize(layers.size());
	v_bias.resize(layers.size());
	m_corrected.resize(layers.size());
	m_bias_corrected.resize(layers.size());
	v_corrected.resize(layers.size());
	v_bias_corrected.resize(layers.size());

	for(size_t i = 0; i < layers.size(); i++) {
		// need to make matrix object for params at each layer
		// use the layer weights/bias shapes which should be initialized
		if(!layers[i]->is_trainable) {
			continue;
		}
		std::vector<int> wshape = layers[i]->w.shape();
		int w_rows = wshape[0];
		int w_cols = wshape[1];
		std::vector<std::vector<double>> tempvec(w_rows, std::vector<double>(w_cols, 0));
		matx::Matrix<double> temp(tempvec);
		m[i] = temp;
		std::vector<std::vector<double>> tempvec2(w_rows, std::vector<double>(w_cols, 0));
		matx::Matrix<double> temp2(tempvec2);
		v[i] = temp2;

		std::vector<int> bshape = layers[i]->b.shape();
		int b_rows = bshape[0];
		int b_cols = bshape[1];
		std::vector<std::vector<double>> tempvec3(b_rows, std::vector<double>(b_cols, 0));
		matx::Matrix<double> temp3(tempvec3);
		m_bias[i] = temp3;

		std::vector<std::vector<double>> tempvec4(b_rows, std::vector<double>(b_cols, 0));
		matx::Matrix<double> temp4(tempvec4);
		v_bias[i] = temp4;
	}
}

void Adam::update_params(std::vector<nn::Layer*>& layers) {
	// std::cout << "time is " << time;
	for(size_t i = 0; i < layers.size(); i++) {
		// std::cout << time << " in\n";
		if(!layers[i]->is_trainable) {
			continue;
		}
		double epsilon = 1e-8;
		m[i] = m[i] * beta1 + (layers[i]->dw) * (1 - beta1);
		v[i] = v[i] * beta2 + (layers[i]->dw.apply_elem_wise(square) * (1 - beta2));
		// std::cout << "m,v fine\n";
		double b1 = (1 - std::pow(beta1, time));
		double b2 = (1 - std::pow(beta2, time));
		m_corrected[i] = m[i] / b1;
		v_corrected[i] = v[i] / b2;

		// for(int i = 0; i < v_corrected.size(); i++) {
		// 	matx::Matrix temp = v_corrected[i];
		// 	temp.print_Matrix();
		// 	std::cout << "\n";
		// }
		// std::cout << "m,v corrected fine\n";
		layers[i]->w = layers[i]->w - (m_corrected[i] * alpha) / (v_corrected[i] + epsilon).apply_elem_wise(rt);
		// std::cout << "dw fine\n";

		m_bias[i] = m_bias[i] * beta1 + (layers[i]->db) * (1 - beta1);
		v_bias[i] = v_bias[i] * beta2 + (layers[i]->db.apply_elem_wise(square) * (1 - beta2));
		m_bias_corrected[i] = m_bias[i] / b1;
		v_bias_corrected[i] = v_bias[i] / b2;
		layers[i]->b = layers[i]->b - (m_bias_corrected[i] * alpha) / (v_bias_corrected[i] + epsilon).apply_elem_wise(rt);
		// std::cout << time << " out\n";
	}
	time++;
}

} // namespace nn