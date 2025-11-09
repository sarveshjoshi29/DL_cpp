
// Created by - Sarvesh Joshi
// Date of creation - 23/7/25
// Last updated - 30/9/25
// #if 0
// #include "SymmetricMatrix.hpp"
#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

namespace matx {
// template<typename T> class SymmetricMatrix;

template<typename T> class Matrix {

	protected:
		vector<vector<T>> data;

		//---------------------------------------------------------------------------
	public:
		void resize(int r, int c) {
			data.resize(r, std::vector<T>(c, T{}));
		}

		Matrix() = default;

		Matrix(const vector<vector<T>> data) : data{data} {
		}

		Matrix(int rows, int cols, T value = T{}) {
			vector<vector<T>> temp(rows, vector<T>(cols, value));
			data = temp;
		}

		vector<T>& operator[](int index) {
			return this->data[index];
		}

		const vector<T>& operator[](int index) const {
			return this->data[index];
		}

		size_t size() const {
			return this->data.size();
		}

		void print_Matrix() const {
			cout << "[";
			for(size_t i = 0; i < this->size(); i++) {
				cout << "[";
				for(size_t j = 0; j < (*this)[0].size(); j++) {
					if(j != (*this)[0].size() - 1) {
						cout << data[i][j] << ", ";
					}

					else {
						cout << data[i][j];
					}
				}
				cout << "]";
				if(i + 1 < this->size()) {
					cout << ",\n";
				}

				else {
					cout << "] ";
				}
			}
			cout << "\n";
		}

		vector<int> shape() const {
			vector<int> ans(2, 0);
			ans[0] = this->data.size();
			ans[1] = this->data[0].size();
			return ans;
		}

		Matrix<T> broadcast_to(int r, int c) const {
			std::vector<int> shape = this->shape();
			matx::Matrix<T> ans(this->data);

			if(r != shape[0] && c != shape[1]) {
				throw std::out_of_range("Unable to broadcast " + to_string(shape[0]) + "," + to_string(shape[1]) + " to " + to_string(r) + "," +
										to_string(c) + "\n");
				// std::cerr << "Unable to broadcast " << shape[0] << "," << shape[1] << " to " << r << "," << c << "\n";
				// exit(1);
			}
			if(r == shape[0]) {
				if(c % shape[1]) {
					throw std::out_of_range("Unable to broadcast " + to_string(shape[0]) + "," + to_string(shape[1]) + " to " + to_string(r) + "," +
											to_string(c) + "\n");
					// std::cerr << "Unable to broadcast " << shape[0] << "," << shape[1] << " to " << r << "," << c << "\n";
					// exit(1);
				}

				int times = c / shape[1];
				for(int ct = 1; ct < times; ct++) {
					int start = shape[1] * ct;
					int end = shape[1] * (ct + 1) - 1;
					for(int i = 0; i < r; i++) {
						for(int j = start; j <= end; j++) {
							ans.data[i].push_back(data[i][j - start]);
						}
					}
				}
				return ans;
			}
			if(r % shape[0]) {
				throw std::out_of_range("Unable to broadcast " + to_string(shape[0]) + "," + to_string(shape[1]) + " to " + to_string(r) + "," +
										to_string(c) + "\n");

				// std::cerr << "Unable to broadcast " << shape[0] << "," << shape[1] << " to " << r << "," << c << "\n";
				// exit(1);
			}

			int times = r / shape[0];
			for(int ct = 1; ct < times; ct++) {
				for(int i = 0; i < shape[0]; i++) {
					std::vector<T> temp;
					for(int j = 0; j < c; j++) {
						temp.push_back(this->data[i][j]);
					}
					ans.data.push_back(temp);
				}
			}
			return ans;
		}

		Matrix operator+(const Matrix& other) const {
			int m = this->size();
			int n = data[0].size();
			int p = other.size();
			int q = other[0].size();
			if(m == p && n == q) {
				vector<vector<T>> ans(m, vector<T>(n, T{}));
				for(int i = 0; i < m; i++) {
					for(int j = 0; j < n; j++) {
						ans[i][j] = (*this)[i][j] + other[i][j];
					}
				}

				return Matrix(ans);
			}

			else if(m == p && (max(n, q) % min(n, q) == 0)) {

				if(n > q) {
					Matrix ans = other.broadcast_to(m, n);
					ans = ans + *this;
					return ans;
				}

				else {
					Matrix ans = this->broadcast_to(m, q);
					ans = ans + other;
					return ans;
				}

			}

			else if(n == q && max(m, p) % min(m, p) == 0) {
				Matrix ans;
				if(m > p) {
					Matrix ans = other.broadcast_to(m, n);
					ans = ans + *this;
					return ans;
				}

				else {
					Matrix ans = this->broadcast_to(p, q);
					ans = ans + other;
					return ans;
				}
			}
			vector<int> this_shape = this->shape();
			vector<int> other_shape = other.shape();
			throw std::out_of_range("Unable to broadcast " + to_string(this_shape[0]) + "," + to_string(this_shape[1]) + " to " +
									to_string(other_shape[0]) + "," + to_string(other_shape[1]) + "\n");
			// std::cerr << "Unable to broadcast " << this_shape[0] << "," << this_shape[1] << " and " << other_shape[0] << "," << other_shape[1]
			// 		  << "\n";
			// exit(1);
		}

		//----------------------------------------------------------------------------------
		Matrix operator-(const Matrix& other) const {
			int m = this->size();
			int n = data[0].size();
			int p = other.size();
			int q = other[0].size();
			if(m == p && n == q) {
				vector<vector<T>> ans(m, vector<T>(n, T{}));
				for(int i = 0; i < m; i++) {
					for(int j = 0; j < n; j++) {
						ans[i][j] = (*this)[i][j] - other[i][j];
					}
				}

				return Matrix(ans);
			}

			else if(m == p && (max(n, q) % min(n, q) == 0)) {

				if(n > q) {
					Matrix ans = other.broadcast_to(m, n);
					ans = *this - ans;
					return ans;
				}

				else {
					Matrix ans = this->broadcast_to(m, q);
					ans = ans - other;
					return ans;
				}

			}

			else if(n == q && max(m, p) % min(m, p) == 0) {

				if(m > p) {
					Matrix ans = other.broadcast_to(m, n);
					ans = *this - ans;
					return ans;
				}

				else {
					Matrix ans = this->broadcast_to(p, q);
					ans = ans - other;
					return ans;
				}
			}
			vector<int> this_shape = this->shape();
			vector<int> other_shape = other.shape();
			throw std::out_of_range("Unable to broadcast " + to_string(this_shape[0]) + "," + to_string(this_shape[1]) + " to " +
									to_string(other_shape[0]) + "," + to_string(other_shape[1]) + "\n");
			// std::cerr << "Unable to broadcast " << this_shape[0] << "," << this_shape[1] << " and " << other_shape[0] << "," << other_shape[1]
			// 		  << "\n";
			// exit(1);
		}

		//-------------------------------------------------------------------------------
		Matrix operator*(const Matrix& other) const {
			int m = this->size();
			int n = data[0].size();
			int p = other.size();
			int q = other[0].size();
			vector<int> this_shape = this->shape();
			vector<int> other_shape = other.shape();
			// std::cerr << "STARTING " << this_shape[0] << "," << this_shape[1] << " and " << other_shape[0] << "," << other_shape[1] << "\n";

			if(m == p && n == q) {
				vector<vector<T>> ans(m, vector<T>(n, T{}));
				for(int i = 0; i < m; i++) {
					for(int j = 0; j < n; j++) {
						ans[i][j] = (*this)[i][j] * other[i][j];
					}
				}

				return Matrix(ans);
			}

			else if(m == p && (max(n, q) % min(n, q) == 0)) {

				if(n > q) {

					Matrix ans = other.broadcast_to(m, n);
					ans = ans * (*this);
					return ans;
				}

				else {
					Matrix ans = this->broadcast_to(m, q);
					ans = ans * other;
					return ans;
				}

			}

			else if(n == q && max(m, p) % min(m, p) == 0) {

				if(m > p) {

					Matrix ans = other.broadcast_to(m, n);
					ans = ans * (*this);
					return ans;
				}

				else {

					Matrix ans = this->broadcast_to(p, q);
					ans = ans * other;
					return ans;
				}
			}

			throw std::out_of_range("Unable to broadcast " + to_string(this_shape[0]) + "," + to_string(this_shape[1]) + " to " +
									to_string(other_shape[0]) + "," + to_string(other_shape[1]) + "\n");
			// std::cerr << "Unable to broadcast " << this_shape[0] << "," << this_shape[1] << " and " << other_shape[0] << "," << other_shape[1]
			// 		  << "\n";
			// exit(1);
		}

		//-------------------------------------------------------------------------------

		Matrix operator/(const Matrix& other) const {
			int m = this->size();
			int n = data[0].size();
			int p = other.size();
			int q = other[0].size();
			if(m == p && n == q) {
				vector<vector<T>> ans(m, vector<T>(n, T{}));
				for(int i = 0; i < m; i++) {
					for(int j = 0; j < n; j++) {
						ans[i][j] = (*this)[i][j] / other[i][j];
					}
				}

				return Matrix(ans);
			}

			else if(m == p && (max(n, q) % min(n, q) == 0)) {

				if(n > q) {
					Matrix ans = other.broadcast_to(m, n);
					ans = *this / ans;
					return ans;
				}

				else {
					Matrix ans = this->broadcast_to(m, q);
					ans = ans / other;
					return ans;
				}

			}

			else if(n == q && max(m, p) % min(m, p) == 0) {

				if(m > p) {
					Matrix ans = other.broadcast_to(m, n);
					ans = *this / ans;
					return ans;
				}

				else {
					Matrix ans = this->broadcast_to(p, q);
					ans = ans / other;
					return ans;
				}
			}
			vector<int> this_shape = this->shape();
			vector<int> other_shape = other.shape();
			throw std::out_of_range("Unable to broadcast " + to_string(this_shape[0]) + "," + to_string(this_shape[1]) + " to " +
									to_string(other_shape[0]) + "," + to_string(other_shape[1]) + "\n");
			// std::cerr << "Unable to broadcast " << this_shape[0] << "," << this_shape[1] << " and " << other_shape[0] << "," << other_shape[1]
			// 		  << "\n";
			// exit(1);
		}

		//-------------------------------------------------------------------------------

		// Matrix multiplication
		Matrix operator%(const Matrix& other) const {

			vector<vector<T>> ans(data.size(), vector<T>(other[0].size(), T{}));
			if(data[0].size() != other.size()) {
				throw std::invalid_argument("Incompatible matrices to multiply of dimensions " + to_string(data.size()) + "x" +
											to_string(data[0].size()) + " " + to_string(other.size()) + "x" + to_string(other[0].size()));
				// cout << "Incompatible matrices to multiply of dimensions " << data.size() << "x" << data[0].size() << " " << other.size() << "x"
				// 	 << other[0].size();
			}
			for(size_t i = 0; i < data.size(); i++) {
				for(size_t k = 0; k < other[0].size(); k++) {
					for(size_t j = 0; j < other.size(); j++) {
						ans[i][k] += data[i][j] * other[j][k];
					}
				}
			}
			return Matrix<T>(ans);
		}

		//----------------------------------------------------------------------------------------------------

		Matrix operator+(const T& other) const {
			Matrix<T> temp(data.size(), data[0].size(), other);
			return *this + temp;
		}

		//------------------------------------------------------------
		// Overloading () operator for slicing

		// row_end and col_end are EXCLUDED from the slice

		Matrix operator()(int row_start, int row_end, int col_start, int col_end) const {
			if(row_end < row_start || col_end < col_start) {
				throw std::invalid_argument("Invalid slice positions entered\n");
			} else if(row_start < 0 || row_end > this->shape()[0] || col_start < 0 || col_end > this->shape()[1]) {
				throw std::invalid_argument("Invalid slice positions entered\n");
			}
			vector<vector<T>> ans(row_end - row_start, vector<T>(col_end - col_start, T{}));
			int ctrow{};
			for(int i = row_start; i < row_end; i++) {
				int ctcol{};
				for(int j = col_start; j < col_end; j++) {
					ans[ctrow][ctcol] = this->data[i][j];
					ctcol++;
				}
				ctrow++;
			}
			return Matrix(ans);
		}

		//------------------------------------------------------------
		pair<int, int> largest_offdiag_elem() const {
			double maxi = -1;
			pair<int, int> coords = make_pair((int)0, (int)0);
			for(int i = 0; i < data.size(); i++) {
				for(int j = 0; j < data[0].size(); j++) {
					if(i != j) {
						coords = (maxi < abs(data[i][j])) ? make_pair(i, j) : coords;
						maxi = (maxi < abs(data[i][j])) ? abs(data[i][j]) : maxi;
					}
				}
			}
			return coords;
		}

		//-------------------------------------------------------------------------------

		Matrix transpose() const {
			vector<vector<T>> ans(data[0].size(), vector<T>(data.size(), T{}));
			for(size_t i = 0; i < data.size(); i++) {
				for(size_t j = 0; j < data[0].size(); j++) {
					ans[j][i] = data[i][j];
				}
			}

			return Matrix(ans);
		}

		//-------------------------------------------------------------------------------------------
		Matrix operator*(const double factor) const {
			matx::Matrix<double> ans(data);
			for(int i = 0; i < ans.shape()[0]; i++) {
				for(int j = 0; j < ans.shape()[1]; j++) {

					ans.data[i][j] *= factor;
				}
			}
			return ans;
		}

		//-------------------------------------------------------------------------------------------------------------------
		Matrix apply_elem_wise(T (*func)(T)) const {
			matx::Matrix<double> ans(data);
			for(int i = 0; i < ans.shape()[0]; i++) {
				for(int j = 0; j < ans.shape()[1]; j++) {
					ans.data[i][j] = func(ans.data[i][j]);
				}
			}
			return ans;
		}

		//---------------------------------------------------------------------------------------------------------
		// required to check  convergence to a diag matrix when compute_eigen is called for a symmetric matrix
		bool is_diagonal() const {
			for(int i = 0; i < this->data.size(); i++) {
				for(int j = 0; j < this->data[0].size(); j++) {
					if(i != j && abs(this->data[i][j]) > 1e-6)
						return false;
				}
			}
			return true;
		}

		// matx::SymmetricMatrix<T> to_symmetric() {
		// 	matx::SymmetricMatrix<T> sym = matx::SymmetricMatrix<T>(this->data);
		// 	return sym;
		// }

		Matrix<double> round_off() {
			for(int i = 0; i < data.size(); i++) {
				for(int j = 0; j < data[0].size(); j++) {
					if(abs(data[i][j] - (int)data[i][j]) < 1e-5) {
						data[i][j] = (int)data[i][j];
						// cout << "ok " << (int)data[i][j];
					}
				}
			}
			return Matrix<double>(data);
		}
};
} // namespace matx

// #endif
