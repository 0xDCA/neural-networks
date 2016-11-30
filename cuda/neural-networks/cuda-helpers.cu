/*
 ============================================================================
 Name        : neural-networks.cu
 Author      :
 Version     :
 Copyright   :
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <assert.h>
#include "ManagedMatrix.h"
#include "cuda-helpers.h"
#include "int-util.h"
#include "TrainSettings.h"

static void checkCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) checkCudaErrorAux(__FILE__,__LINE__, #value, value)

namespace dc_internal {
	struct IterationSettings {
		int activations_size;
		int weights_size;
		double regularization_term;
		double step_factor;
		double momentum;
		int inner_steps;
		long long seed;
	};

	__device__ __host__ double sigmoid(double t) {
	    return 1.0 / (1 + exp(-t));
	}

	/**
	 * Multiplies the matrix A by the vector x, and stores the result in y.
	 * The matrix A must be stored in column-wise order. Elements of y will
	 * be initialized to zero
	 *
	 * 'A' has dimensions rows * cols.
	 * 'x' is a 'cols'-length vector.
	 * 'y' is a 'rows'-length vector.
	 *
	 */
	__device__ void multiply_matrix_vector(int rows, int cols, const double* a, const double* x, double* y) {
		for(int i = 0; i < rows; ++i) {
			y[i] = 0;
			for(int j = 0; j < cols; ++j) {
				y[i] += x[j] * a[rows * j + i];
			}
		}
	}

	/**
	 * Multiplies the vector x by the transpose of vector y, and stores the result in A.
	 * Elements of A WILL NOT be initialized to zero.
	 *
	 * 'A' has dimensions rows * cols.
	 * 'x' is a 'rows'-length vector.
	 * 'y' is a 'cols'-length vector.
	 *
	 */
	__device__ void multiply_vector_vector_t_no_init(int rows, int cols, const double* x, const double* y, double* a) {
		for(int i = 0; i < rows; ++i) {
			for(int j = 0; j < cols; ++j) {
				a[rows * j + i] += x[i] * y[j];
			}
		}
	}

	/**
	 * Subtracts the vectors 'a' and 'b' (of length 'n'), and stores the result in 'out' (of length 'n').
	 */
	__device__ void subtract_vectors(int n, const double* a, const double* b, double* out) {
		for(int i = 0; i < n; ++i) {
			out[i] = a[i] - b[i];
		}
	}

	__device__ void forward_propagation(int layers, const int* layer_neurons,
			const double* input, const double* merged_weights, double* merged_out) {
		double *x = merged_out;

		// Copy the activations for the input layer
		*merged_out = 1;
		++merged_out;

		memcpy(merged_out, input, sizeof(*input) * (*layer_neurons));
		merged_out += *layer_neurons;

		// Compute the rest of activations
		for(int i = 0; i < layers - 2; ++i) {
			*merged_out = 1;
			multiply_matrix_vector(layer_neurons[i + 1], layer_neurons[i] + 1, merged_weights, x, merged_out + 1);

			x = merged_out;
			++merged_out;

			for (int j = 0; j < layer_neurons[i + 1]; ++j) {
				*merged_out = sigmoid(*merged_out);
				++merged_out;
			}

			merged_weights += layer_neurons[i + 1] * (layer_neurons[i] + 1);
		}

		multiply_matrix_vector(layer_neurons[layers - 1], layer_neurons[layers - 2] + 1,
				merged_weights, x, merged_out);
		for (int j = 0; j < layer_neurons[layers - 1]; ++j) {
			*merged_out = sigmoid(*merged_out);
			++merged_out;
		}

	}

	__global__ void forward_propagation_kernel(int layers, const int* layer_neurons,
				const double* input, const double* merged_weights, double* merged_out) {
		forward_propagation(layers, layer_neurons, input, merged_weights, merged_out);
	}

	__device__ void back_propagation(int layers, const int* layer_neurons,
			const double* merged_activations, const double* y,
			const double* merged_weights, double* merged_out) {

		int max_layer_size = layer_neurons[layers - 1];
		// Move pointers to the last layer
		for(int i = 0; i < layers - 1; ++i) {
			max_layer_size = max(max_layer_size, layer_neurons[i]);
			merged_activations += layer_neurons[i] + 1;

			merged_out += layer_neurons[i + 1] * (layer_neurons[i] + 1);
			merged_weights += layer_neurons[i + 1] * (layer_neurons[i] + 1);
		}

		double* delta_next = (double*) malloc(sizeof(double) * max_layer_size);
		double* delta_next_copy = (double*) malloc(sizeof(double) * max_layer_size);
		assert(delta_next);
		assert(delta_next_copy);

		// Compute gradient for the last layer
		subtract_vectors(layer_neurons[layers - 1], merged_activations, y, delta_next);

		for(int i = layers - 2; i >= 0; --i) {
			merged_activations -= layer_neurons[i] + 1;
			merged_out -= layer_neurons[i + 1] * (layer_neurons[i] + 1);
			merged_weights -= layer_neurons[i + 1] * (layer_neurons[i] + 1);
			multiply_vector_vector_t_no_init(layer_neurons[i + 1], layer_neurons[i] + 1,
					delta_next, merged_activations, merged_out);

			for(int j = 0; j < layer_neurons[i]; ++j) {
				int cols_t = layer_neurons[i + 1];
				for (int k = 0; k < cols_t; ++k) {
					// Omit the first row using (j + 1) instead of j
					delta_next_copy[j] = merged_weights[cols_t * (j + 1) + k] * delta_next[k] *
							merged_activations[j + 1] * (1.0 - merged_activations[j + 1]);
				}
			}

			double* temp = delta_next;
			delta_next = delta_next_copy;
			delta_next_copy = temp;
		}

		free(delta_next);
		free(delta_next_copy);
	}

	__device__ void print_weights(int layers, const int* layer_neurons, const double* weights) {
		for(int i = 0; i < layers - 1; ++i) {
			printf("Layer %d to %d:\n", i, i + 1);
			for(int j = 0; j < layer_neurons[i + 1]; ++j) {
				for(int k = 0; k < layer_neurons[i] + 1; ++k) {
					printf("%.10lf ", weights[k * layer_neurons[i + 1] + j]);
				}
				printf("\n");
			}

			weights += layer_neurons[i + 1] * (layer_neurons[i] + 1);
		}
	}

	__global__ void compute_error_kernel(int examples_per_thread,
			double regularization_term, int activations_size, int layers,
			const int* layer_neurons, const double* merged_weights, int examples, const double* x,
			const double* y, double* result) {
		int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
		int start = examples_per_thread * thread_id;
		int end = min(start + examples_per_thread, examples);

		if (thread_id == 0) {
			double complexity_term = 0;
			const double* temp_merged_weights = merged_weights;
			for (int i = 0; i < layers - 1; ++i) {
				int dimension = layer_neurons[i + 1] * (layer_neurons[i] + 1);
				for (int j = 0; j < dimension; ++j) {
					complexity_term += temp_merged_weights[j] * temp_merged_weights[j];
				}

				temp_merged_weights += dimension;
			}

			atomicAdd(result, regularization_term / (2.0 * examples) * complexity_term);
		}

		if (start < end) {
			double error = 0;

			double* activations = (double*) malloc(sizeof(double) * activations_size);
			assert(activations);

			for(int i = start; i < end; ++i) {
				const double* xi_pointer = x + layer_neurons[0] * i;

				forward_propagation(layers, layer_neurons, xi_pointer, merged_weights, activations);

				double norm = 0;
				for(int j = 0; j < layer_neurons[layers - 1]; ++j) {
					double temp = y[layer_neurons[layers - 1] * i + j] -
							activations[activations_size - layer_neurons[layers - 1] + j];
					norm += temp * temp;
				}

				error += norm;
			}


			free(activations);

			atomicAdd(result, 1.0 / examples * error);
		}
	}

	__global__ void back_propagation_kernel(int layers, const int* layer_neurons,
			const double* merged_activations, const double* y,
			const double* merged_weights, double* merged_out) {
		back_propagation(layers, layer_neurons, merged_activations, y, merged_weights, merged_out);
	}

	__global__ void train_kernel(IterationSettings settings,
			int layers, const int* layer_neurons, int examples, const double* x, const double* y,
			const double* master_weights, double* thread_weights) {
		int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
		thread_weights += settings.weights_size * thread_id;

		// Copy from master_weights
		memcpy(thread_weights, master_weights, sizeof(double) * settings.weights_size);

		double* deltas = (double*) malloc(sizeof(double) * settings.weights_size);
		assert(deltas);

		double* current_deltas = (double*) malloc(sizeof(double) * settings.weights_size);
		assert(current_deltas);

		double* activations = (double*) malloc(sizeof(double) * settings.activations_size);
		assert(activations);

		curandState state;
		curand_init(settings.seed, thread_id, 0, &state);

		// Initialize deltas to 0
		for(int i = 0; i < layers - 1; ++i) {
			int dimensions = layer_neurons[i + 1] * (layer_neurons[i] + 1);
			for(int j = 0; j < dimensions; ++j) {
				deltas[j] = 0;
			}

			deltas += dimensions;
		}

		deltas -= settings.weights_size;

		for (int t = 0; t < settings.inner_steps; ++t) {
			for(int i = 0; i < layers - 1; ++i) {
				int dimensions = layer_neurons[i + 1] * (layer_neurons[i] + 1);
				for(int j = 0; j < dimensions; ++j) {
					current_deltas[j] = 0;
				}

				current_deltas += dimensions;
			}
			current_deltas -= settings.weights_size;

			int example = static_cast<int>(round(curand_uniform(&state) * (examples - 1)));

			/*printf("t = %d. Example: %d. Weights:\n", t, example);
			print_weights(layers, layer_neurons, thread_weights);*/

			forward_propagation(layers, layer_neurons, &x[example * layer_neurons[0]],
					thread_weights, activations);
			back_propagation(layers, layer_neurons, activations,
					&y[example * layer_neurons[layers - 1]], thread_weights, current_deltas);

			for(int i = 0; i < layers - 1; ++i) {
				for(int j = 0; j < layer_neurons[i + 1]; ++j) {
					for(int k = 0; k < layer_neurons[i] + 1; ++k) {
						int index = k * layer_neurons[i + 1] + j;
						double denominator = examples;

						current_deltas[index] /= denominator;
						if (k > 0) {
							current_deltas[index] += settings.regularization_term / denominator * thread_weights[index];
						}

						deltas[index] = settings.momentum * deltas[index] + current_deltas[index];

						thread_weights[index] -= settings.step_factor * deltas[index];
					}
				}

				int dimensions = layer_neurons[i + 1] * (layer_neurons[i] + 1);
				current_deltas += dimensions;
				deltas += dimensions;
				thread_weights += dimensions;
			}
			current_deltas -= settings.weights_size;
			deltas -= settings.weights_size;
			thread_weights -= settings.weights_size;
		}

		free(activations);
		free(current_deltas);
		free(deltas);
	}

	__global__ void set_weights_to_zero_kernel(int elements_per_thread,
			int weights_size, double* master_weights) {
		int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
		int start = thread_id * elements_per_thread;
		int end = min(start + elements_per_thread, weights_size);

		while (start < end) {
			master_weights[start] = 0;
			++start;
		}
	}

	__global__ void merge_weights_kernel(int weights_size, int indexes_per_thread,
			const double* thread_weights, double* master_weights) {
		int total_threads = gridDim.x * blockDim.x;
		int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
		int start = indexes_per_thread * thread_id;
		int end = min(start + indexes_per_thread, weights_size * total_threads);

		for(int j = start; j < end; ++j) {
			double original = master_weights[j];
			//master_weights[j] = 0;
			for(int i = 0; i < total_threads; ++i) {
				master_weights[j] += (thread_weights[i * weights_size + j] - original) / total_threads;
			}
		}
	}

	template <class T>
	inline T* copy_vector_to_device(const std::vector<T>& source) {
		T* result;
		CUDA_CHECK_RETURN(cudaMalloc(&result, sizeof(T) * source.size()));
		CUDA_CHECK_RETURN(cudaMemcpy(result, source.data(), sizeof(T) * source.size(), cudaMemcpyHostToDevice));

		return result;
	}

	template <class T>
	inline T* copy_matrix_to_device(const dc::ManagedMatrix<T>& source) {
		T* result;
		int dimensions = source.get_columns() * source.get_rows();

		CUDA_CHECK_RETURN(cudaMalloc(&result, sizeof(T) * dimensions));
		CUDA_CHECK_RETURN(cudaMemcpy(result, source.get_internal_data(),
				sizeof(T) * dimensions, cudaMemcpyHostToDevice));

		return result;
	}

	template <class T>
	inline T* copy_merged_matrices_to_device(const std::vector<dc::ManagedMatrix<T> >& source) {
		int total_size = 0;
		for(const auto& w : source) {
			total_size += w.get_rows() * w.get_columns();
		}

		T* d_result;
		CUDA_CHECK_RETURN(cudaMalloc(&d_result, sizeof(T) * total_size));
		for(auto& w : source) {
			int length = w.get_columns() * w.get_rows();

			CUDA_CHECK_RETURN(cudaMemcpy(d_result, w.get_internal_data(), length * sizeof(T),
					cudaMemcpyHostToDevice));


			d_result += length;
		}

		d_result -= total_size;

		return d_result;
	}

	std::vector<std::vector<double> > forward_propagate(const std::vector<int>& layer_neurons, const std::vector<double>& input,
			const std::vector<dc::ManagedMatrix<double> >& weights) {
		int* d_layer_neurons = dc_internal::copy_vector_to_device(layer_neurons);
		double* d_input = dc_internal::copy_vector_to_device(input);

		double* d_out;
		int out_size = -1; // Subtract additional bias from the output layer.
		for(int n : layer_neurons) {
			out_size += n + 1;
		}

		CUDA_CHECK_RETURN(cudaMalloc(&d_out, sizeof(double) * out_size));

		double* d_weights = copy_merged_matrices_to_device(weights);

		dc_internal::forward_propagation_kernel<<<1,1>>>(
				layer_neurons.size(), d_layer_neurons, d_input, d_weights, d_out);
		CUDA_CHECK_RETURN(cudaGetLastError());

		std::vector<std::vector<double> > result(layer_neurons.size());
		for(int i = 0; i < layer_neurons.size(); ++i) {
			result[i] = std::vector<double>(layer_neurons[i] +
					/* bias */ (i != static_cast<int>(layer_neurons.size()) - 1));

			CUDA_CHECK_RETURN(cudaMemcpy(result[i].data(), d_out, sizeof(double) * result[i].size(), cudaMemcpyDeviceToHost));
			d_out += result[i].size();
		}

		d_out -= out_size;


		CUDA_CHECK_RETURN(cudaFree(d_weights));
		CUDA_CHECK_RETURN(cudaFree(d_out));
		CUDA_CHECK_RETURN(cudaFree(d_input));
		CUDA_CHECK_RETURN(cudaFree(d_layer_neurons));

		return result;
	}

	std::vector<dc::ManagedMatrix<double> > back_propagate(const std::vector<int>& layer_neurons,
			const std::vector<double>& input, const std::vector<double>& output,
			const std::vector<dc::ManagedMatrix<double> >& weights) {
		int* d_layer_neurons = dc_internal::copy_vector_to_device(layer_neurons);
		double* d_input = dc_internal::copy_vector_to_device(input);
		double* d_output = dc_internal::copy_vector_to_device(output);

		double* d_activations;
		int activations_size = -1; // Subtract additional bias from the output layer.
		for(int n : layer_neurons) {
			activations_size += n + 1;
		}

		CUDA_CHECK_RETURN(cudaMalloc(&d_activations, sizeof(double) * activations_size));

		double* d_weights = copy_merged_matrices_to_device(weights);

		std::vector<dc::ManagedMatrix<double> > result;
		for(int i = 0; i < weights.size(); ++i) {
			result.push_back(dc::ManagedMatrix<double>(weights[i].get_rows(), weights[i].get_columns(), 0.0));
		}
		double* d_out = dc_internal::copy_merged_matrices_to_device(result);

		dc_internal::forward_propagation_kernel<<<1,1>>>(
				layer_neurons.size(), d_layer_neurons, d_input, d_weights, d_activations);
		CUDA_CHECK_RETURN(cudaGetLastError());
		dc_internal::back_propagation_kernel<<<1,1>>>(
				layer_neurons.size(), d_layer_neurons, d_activations, d_output, d_weights, d_out);
		CUDA_CHECK_RETURN(cudaGetLastError());

		double* d_out_temp = d_out;
		for(int i = 0; i < result.size(); ++i) {
			int dimensions = result[i].get_columns() * result[i].get_rows();
			CUDA_CHECK_RETURN(cudaMemcpy(result[i].get_internal_data_unsafe(), d_out_temp,
					sizeof(double) * dimensions, cudaMemcpyDeviceToHost));
			d_out_temp += dimensions;
		}

		CUDA_CHECK_RETURN(cudaFree(d_out));
		CUDA_CHECK_RETURN(cudaFree(d_weights));
		CUDA_CHECK_RETURN(cudaFree(d_activations));
		CUDA_CHECK_RETURN(cudaFree(d_output));
		CUDA_CHECK_RETURN(cudaFree(d_input));
		CUDA_CHECK_RETURN(cudaFree(d_layer_neurons));

		return result;
	}

	TrainResult train(const std::vector<int>& layer_neurons,
			const dc::ManagedMatrix<double>& x,
			const dc::ManagedMatrix<double>& y, const TrainSettings& train_settings,
			std::vector<dc::ManagedMatrix<double> >& weights) {
		int* d_layer_neurons = dc_internal::copy_vector_to_device(layer_neurons);
		double* d_x = copy_matrix_to_device(x);
		double* d_y = copy_matrix_to_device(y);

		int activations_size = -1; // Subtract additional bias from the output layer.
		for(int n : layer_neurons) {
			activations_size += n + 1;
		}

		double* d_weights = copy_merged_matrices_to_device(weights);

		const int blocks = train_settings.blocks;
		const int threads_per_block = int_division_round_up(train_settings.threads, blocks);
		const int actual_threads = blocks * threads_per_block;

		int weights_size = 0;
		for(const auto& w : weights) {
			weights_size += w.get_rows() * w.get_columns();
		}

		double* d_thread_weights;
		CUDA_CHECK_RETURN(cudaMalloc(&d_thread_weights, sizeof(double) * weights_size * actual_threads));

		IterationSettings settings;
		settings.activations_size = activations_size;
		settings.weights_size = weights_size;
		settings.inner_steps = train_settings.inner_steps;
		settings.momentum = train_settings.momentum;
		settings.regularization_term = train_settings.regularization_term;
		settings.seed = time(NULL);
		settings.step_factor = train_settings.step_factor;

		int examples = x.get_columns();
		int examples_per_thread = int_division_round_up(x.get_columns(), actual_threads);
		double error = 0;
		double* d_error;
		CUDA_CHECK_RETURN(cudaMalloc(&d_error, sizeof(double)));

		CUDA_CHECK_RETURN(cudaMemcpy(d_error, &error, sizeof(double), cudaMemcpyHostToDevice));
		compute_error_kernel<<<blocks, threads_per_block>>>(examples, train_settings.regularization_term,
				activations_size, layer_neurons.size(), d_layer_neurons, d_weights, examples, d_x, d_y,
				d_error);
		CUDA_CHECK_RETURN(cudaGetLastError());
		CUDA_CHECK_RETURN(cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost));

		int it;
		for(it = 1; it <= train_settings.iterations; ++it) {
			if (error <= train_settings.target_error) {
				break;
			}

			train_kernel<<<blocks, threads_per_block>>>(settings, layer_neurons.size(),
					d_layer_neurons, examples, d_x, d_y, d_weights, d_thread_weights);
			CUDA_CHECK_RETURN(cudaGetLastError());

			int indexes_per_thread = int_division_round_up(weights_size, actual_threads);
			merge_weights_kernel<<<blocks, threads_per_block>>>(weights_size, indexes_per_thread,
					d_thread_weights, d_weights);
			CUDA_CHECK_RETURN(cudaGetLastError());

			error = 0;
			CUDA_CHECK_RETURN(cudaMemcpy(d_error, &error, sizeof(double), cudaMemcpyHostToDevice));
			compute_error_kernel<<<blocks, threads_per_block>>>(examples_per_thread, train_settings.regularization_term,
							activations_size, layer_neurons.size(), d_layer_neurons, d_weights, examples, d_x, d_y,
							d_error);
			CUDA_CHECK_RETURN(cudaGetLastError());
			CUDA_CHECK_RETURN(cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost));

			//std::cout << "Iteration " << it << ". Error: " << error << std::endl;
		}

		for(int i = 0; i < weights.size(); ++i) {
			int dimensions = weights[i].get_columns() * weights[i].get_rows();
			CUDA_CHECK_RETURN(cudaMemcpy(weights[i].get_internal_data_unsafe(),
					d_weights, sizeof(double) * dimensions, cudaMemcpyDeviceToHost));
			d_weights += dimensions;
		}

		d_weights -= weights_size;

		CUDA_CHECK_RETURN(cudaFree(d_error));
		CUDA_CHECK_RETURN(cudaFree(d_thread_weights));
		CUDA_CHECK_RETURN(cudaFree(d_weights));
		CUDA_CHECK_RETURN(cudaFree(d_y));
		CUDA_CHECK_RETURN(cudaFree(d_x));
		CUDA_CHECK_RETURN(cudaFree(d_layer_neurons));

		return TrainResult(min(it, train_settings.iterations), error);
	}

	double compute_error(const std::vector<int>& layer_neurons,
				const dc::ManagedMatrix<double>& x,
				const dc::ManagedMatrix<double>& y,
				std::vector<dc::ManagedMatrix<double> >& weights,
				double regularization_term, int blocks, int threads_per_block) {
		int* d_layer_neurons = dc_internal::copy_vector_to_device(layer_neurons);
		double* d_x = copy_matrix_to_device(x);
		double* d_y = copy_matrix_to_device(y);

		double* d_weights = copy_merged_matrices_to_device(weights);

		double error = 0;
		double* d_error;
		CUDA_CHECK_RETURN(cudaMalloc(&d_error, sizeof(double)));
		CUDA_CHECK_RETURN(cudaMemcpy(d_error, &error, sizeof(double), cudaMemcpyHostToDevice));

		int actual_threads = blocks * threads_per_block;
		int examples = x.get_columns();
		int examples_per_thread = int_division_round_up(x.get_columns(), actual_threads);

		int activations_size = -1; // Subtract additional bias from the output layer.
		for(int n : layer_neurons) {
			activations_size += n + 1;
		}

		compute_error_kernel<<<blocks, threads_per_block>>>(examples_per_thread, regularization_term,
						activations_size, layer_neurons.size(), d_layer_neurons, d_weights, examples, d_x, d_y,
						d_error);
		CUDA_CHECK_RETURN(cudaGetLastError());
		CUDA_CHECK_RETURN(cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost));

		CUDA_CHECK_RETURN(cudaFree(d_error));
		CUDA_CHECK_RETURN(cudaFree(d_weights));
		CUDA_CHECK_RETURN(cudaFree(d_y));
		CUDA_CHECK_RETURN(cudaFree(d_x));
		CUDA_CHECK_RETURN(cudaFree(d_layer_neurons));

		return error;
	}
}


/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void checkCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess) {
		return;
	}

	std::cerr << statement <<" returned " << cudaGetErrorString(err) << "(" << err << ") at "
			<< file << ":" << line << std::endl;
	exit(1);
}
