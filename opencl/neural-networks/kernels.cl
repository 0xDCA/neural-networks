double sigmoid(double t) {
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
void multiply_matrix_vector(int rows, int cols, const double* a, const double* x, double* y) {
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
void multiply_vector_vector_t_no_init(int rows, int cols, const double* x, const double* y, double* a) {
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < cols; ++j) {
			a[rows * j + i] += x[i] * y[j];
		}
	}
}

/**
 * Subtracts the vectors 'a' and 'b' (of length 'n'), and stores the result in 'out' (of length 'n').
 */
void subtract_vectors(int n, const double* a, const double* b, double* out) {
	for(int i = 0; i < n; ++i) {
		out[i] = a[i] - b[i];
	}
}

void forward_propagation(int layers, const int* layer_neurons,
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

__kernel void forward_propagation_kernel(int layers, const int* layer_neurons,
			const double* input, const double* merged_weights, double* merged_out) {
	forward_propagation(layers, layer_neurons, input, merged_weights, merged_out);
}

void back_propagation(int layers, const int* layer_neurons,
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

void print_weights(int layers, const int* layer_neurons, const double* weights) {
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

__kernel void compute_error_kernel(int examples_per_thread,
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

__kernel void back_propagation_kernel(int layers, const int* layer_neurons,
		const double* merged_activations, const double* y,
		const double* merged_weights, double* merged_out) {
	back_propagation(layers, layer_neurons, merged_activations, y, merged_weights, merged_out);
}

__kernel void train_kernel(IterationSettings settings,
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

__kernel void set_weights_to_zero_kernel(int elements_per_thread,
		int weights_size, double* master_weights) {
	int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	int start = thread_id * elements_per_thread;
	int end = min(start + elements_per_thread, weights_size);

	while (start < end) {
		master_weights[start] = 0;
		++start;
	}
}

__kernel void merge_weights_kernel(int weights_size, int indexes_per_thread,
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
