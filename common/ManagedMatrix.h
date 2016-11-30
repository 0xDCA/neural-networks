/*
 * ManagedMatrix.h
 *
 *  Created on: 7/10/2016
 *      Author: drcastroa
 */

#ifndef MANAGEDMATRIX_H_
#define MANAGEDMATRIX_H_

#include <vector>
#include <iostream>
#include <sstream>
#include <string>

namespace dc {
template <class TElem>
class ManagedMatrix {
public:
	ManagedMatrix(int rows, int columns, const TElem& default_value = TElem()) :
		columns(columns), rows(rows), data(columns * rows, default_value) {}

	int get_columns() const { return columns; }
	int get_rows() const { return rows; }

	const TElem* get_internal_data() const {
		return data.data();
	}

	TElem* get_internal_data_unsafe() {
		return data.data();
	}

	void set_all_row_wise(const std::vector<TElem>& source) {
		for(int i = 0; i < rows; ++i) {
			for(int j = 0; j < columns; ++j) {
				data[j * rows + i] = source[i * columns + j];
			}
		}
	}

	void set_all_column_wise(const std::vector<TElem>& source) {
		data = source;
	}

	void set_all(TElem value) {
		data = std::vector<TElem>(columns * rows, value);
	}

	void set(int row, int col, TElem value) {
		validate_access(row, col);
		data[col * rows + row] = value;
	}

	TElem get(int row, int col) const {
		validate_access(row, col);
		return data[col * rows + row];
	}

	ManagedMatrix<TElem> get_transposed() const {
		ManagedMatrix<TElem> result(columns, rows);

		result.set_all_row_wise(data);

		return result;
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << this;

		return oss.str();
	}

	template <class T>
	friend bool operator==(const dc::ManagedMatrix<T>& a, const dc::ManagedMatrix<T>& b);

private:
	int columns, rows;
	std::vector<TElem> data;

	void validate_access(int row, int col) const {
		if (row < 0 || row > rows || col < 0 || col > columns) {
			throw std::runtime_error("Invalid access");
		}
	}
};

template <class TElem>
bool operator==(const ManagedMatrix<TElem>& a, const ManagedMatrix<TElem>& b) {
	return a.rows == b.rows && a.columns == b.columns && a.data == b.data;
}

template <class TElem>
std::ostream& operator<<(std::ostream& os, const ManagedMatrix<TElem>& matrix)
{
    for(int i = 0; i < matrix.get_rows(); ++i) {
    	for(int j = 0; j < matrix.get_columns(); ++j) {
    		os << matrix.get(i, j) << ' ';
    	}
    	os << '\n';
    }
    return os;
}
}

#endif /* MANAGEDMATRIX_H_ */
