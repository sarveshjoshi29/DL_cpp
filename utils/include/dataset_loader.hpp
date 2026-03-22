#pragma once
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
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace utils {
class csv_parser {
	public:
		std::vector<std::vector<double>> data_X;
		std::vector<std::vector<double>> data_y;

		csv_parser(std::string file_name);
		void parse(std::ifstream& infile);
};
} // namespace utils