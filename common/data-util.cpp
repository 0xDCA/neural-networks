#include "data-util.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <endian.h>

using namespace std;
using namespace dc;

std::pair<dc::ManagedMatrix<double>, dc::ManagedMatrix<double> > generate_data(int n, std::mt19937& generator) {
	dc::ManagedMatrix<double> x_matrix(n, 2);
	dc::ManagedMatrix<double> y_matrix(n, 1);
	std::uniform_real_distribution<> distribution(-1000.0, 1000.0);

	for (int i = 0; i < n; ++i) {
		double x1 = distribution(generator);
		double x2 = distribution(generator);

		double y = x2*x2 + x1 >= 5000 ? 1.0 : 0.0;

		x_matrix.set(i, 0, x1);
		x_matrix.set(i, 1, x2);
		y_matrix.set(i, 0, y);
	}

	return std::make_pair(x_matrix, y_matrix);
}

std::pair<dc::ManagedMatrix<double>, dc::ManagedMatrix<double> > read_mnist_database(
		const std::string& image_file_name, const std::string& label_file_name) {
	ifstream image_data (image_file_name, ios::in | ios::binary);
	ifstream label_data(label_file_name, ios::in | ios::binary);

	image_data.seekg(0x04);

	int examples = 0, rows = 0, cols = 0;

	image_data.read((char*) &examples, 4);
	image_data.read((char*) &rows, 4);
	image_data.read((char*) &cols, 4);

	examples = be32toh(examples);
	rows = be32toh(rows);
	cols = be32toh(cols);

	int dimensions = rows * cols;

	cout << "Reading " << examples << " examples (" << rows << " x " << cols << ")\n";

	ManagedMatrix<double> x(dimensions, examples);
	for (int i = 0; i < examples; ++i) {
		for (int j = 0; j < dimensions; ++j) {
			unsigned char pixel;
			image_data.read((char*) &pixel, 1);

			x.set(j, i, pixel);
		}
	}

	ManagedMatrix<double> y(1, examples);
	label_data.seekg(0x08);
	for (int i = 0; i < examples; ++i) {
		unsigned char label;
		label_data.read((char*) &label, 1);

		y.set(0, i, label);
	}

	cout << "Done reading\n";

	return make_pair(x, y);
}

std::pair<dc::ManagedMatrix<double>, dc::ManagedMatrix<double> > read_iris_database(
		const std::string& file_name) {
  const int examples = 150;
  const int input_columns = 3;
  const int output_columns = 1;

  cout << "Reading " << examples << " examples\n";

  ifstream data(file_name);

  ManagedMatrix<double> x(input_columns, examples);
  ManagedMatrix<double> y(output_columns, examples);

  for(int i = 0; i < examples; ++i) {
    for(int j = 0; j < input_columns; ++j) {
      double value;
      data >> value;

      x.set(j, i, value);
    }

    for(int j = 0; j < output_columns; ++j) {
      double value;
      data >> value;

      y.set(j, i, value);
    }
  }

  cout << "Done reading\n";

  return make_pair(x, y);
}
