#pragma once
#include "Matrix.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
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

namespace utils {
class file_manager {
	public:
		file_manager();

		template<typename T> void save(std::string file_name, const matx::Matrix<T>& data) {
			std::ofstream out(file_name);
			for(int i = 0; i < data.size(); i++) {
				for(int j = 0; j < data[0].size(); j++) {
					out << data[i][j] << ",";
				}
				out << "\n";
			}
			out.close();
		}
};
} // namespace utils
