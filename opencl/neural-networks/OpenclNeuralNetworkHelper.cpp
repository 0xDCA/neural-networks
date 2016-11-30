#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <assert.h>
#include "ManagedMatrix.h"
#include "OpenclNeuralNetworkHelper.h"
#include "int-util.h"
#include "TrainSettings.h"

static void checkOpenclErrorAux (const char *, unsigned, const char *, cl_int);
#define OPENCL_CHECK_RETURN(value) checkOpenclErrorAux(__FILE__,__LINE__, #value, value)

struct IterationSettings {
	int activations_size;
	int weights_size;
	double regularization_term;
	double step_factor;
	double momentum;
	int inner_steps;
	long long seed;
};

template <class T>
inline T* copy_vector_to_device(const std::vector<T>& source) {
	T* result;
	OPENCL_CHECK_RETURN(cudaMalloc(&result, sizeof(T) * source.size()));
	OPENCL_CHECK_RETURN(cudaMemcpy(result, source.data(), sizeof(T) * source.size(), cudaMemcpyHostToDevice));

	return result;
}

template <class T>
inline T* copy_matrix_to_device(const dc::ManagedMatrix<T>& source) {
	T* result;
	int dimensions = source.get_columns() * source.get_rows();

	OPENCL_CHECK_RETURN(cudaMalloc(&result, sizeof(T) * dimensions));
	OPENCL_CHECK_RETURN(cudaMemcpy(result, source.get_internal_data(),
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
	OPENCL_CHECK_RETURN(cudaMalloc(&d_result, sizeof(T) * total_size));
	for(auto& w : source) {
		int length = w.get_columns() * w.get_rows();

		OPENCL_CHECK_RETURN(cudaMemcpy(d_result, w.get_internal_data(), length * sizeof(T),
				cudaMemcpyHostToDevice));


		d_result += length;
	}

	d_result -= total_size;

	return d_result;
}

std::vector<std::vector<double> > OpenclNeuralNetworkHelper::forward_propagate(const std::vector<int>& layer_neurons, const std::vector<double>& input,
		const std::vector<dc::ManagedMatrix<double> >& weights) {
	int* d_layer_neurons = dc_internal::copy_vector_to_device(layer_neurons);
	double* d_input = dc_internal::copy_vector_to_device(input);

	double* d_out;
	int out_size = -1; // Subtract additional bias from the output layer.
	for(int n : layer_neurons) {
		out_size += n + 1;
	}

	OPENCL_CHECK_RETURN(cudaMalloc(&d_out, sizeof(double) * out_size));

	double* d_weights = copy_merged_matrices_to_device(weights);

	dc_internal::forward_propagation_kernel<<<1,1>>>(
			layer_neurons.size(), d_layer_neurons, d_input, d_weights, d_out);
	OPENCL_CHECK_RETURN(cudaGetLastError());

	std::vector<std::vector<double> > result(layer_neurons.size());
	for(int i = 0; i < layer_neurons.size(); ++i) {
		result[i] = std::vector<double>(layer_neurons[i] +
				/* bias */ (i != static_cast<int>(layer_neurons.size()) - 1));

		OPENCL_CHECK_RETURN(cudaMemcpy(result[i].data(), d_out, sizeof(double) * result[i].size(), cudaMemcpyDeviceToHost));
		d_out += result[i].size();
	}

	d_out -= out_size;


	OPENCL_CHECK_RETURN(cudaFree(d_weights));
	OPENCL_CHECK_RETURN(cudaFree(d_out));
	OPENCL_CHECK_RETURN(cudaFree(d_input));
	OPENCL_CHECK_RETURN(cudaFree(d_layer_neurons));

	return result;
}

std::vector<dc::ManagedMatrix<double> > OpenclNeuralNetworkHelper::back_propagate(const std::vector<int>& layer_neurons,
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

	OPENCL_CHECK_RETURN(cudaMalloc(&d_activations, sizeof(double) * activations_size));

	double* d_weights = copy_merged_matrices_to_device(weights);

	std::vector<dc::ManagedMatrix<double> > result;
	for(int i = 0; i < weights.size(); ++i) {
		result.push_back(dc::ManagedMatrix<double>(weights[i].get_rows(), weights[i].get_columns(), 0.0));
	}
	double* d_out = dc_internal::copy_merged_matrices_to_device(result);

	dc_internal::forward_propagation_kernel<<<1,1>>>(
			layer_neurons.size(), d_layer_neurons, d_input, d_weights, d_activations);
	OPENCL_CHECK_RETURN(cudaGetLastError());
	dc_internal::back_propagation_kernel<<<1,1>>>(
			layer_neurons.size(), d_layer_neurons, d_activations, d_output, d_weights, d_out);
	OPENCL_CHECK_RETURN(cudaGetLastError());

	double* d_out_temp = d_out;
	for(int i = 0; i < result.size(); ++i) {
		int dimensions = result[i].get_columns() * result[i].get_rows();
		OPENCL_CHECK_RETURN(cudaMemcpy(result[i].get_internal_data_unsafe(), d_out_temp,
				sizeof(double) * dimensions, cudaMemcpyDeviceToHost));
		d_out_temp += dimensions;
	}

	OPENCL_CHECK_RETURN(cudaFree(d_out));
	OPENCL_CHECK_RETURN(cudaFree(d_weights));
	OPENCL_CHECK_RETURN(cudaFree(d_activations));
	OPENCL_CHECK_RETURN(cudaFree(d_output));
	OPENCL_CHECK_RETURN(cudaFree(d_input));
	OPENCL_CHECK_RETURN(cudaFree(d_layer_neurons));

	return result;
}

TrainResult OpenclNeuralNetworkHelper::train(const std::vector<int>& layer_neurons,
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
	OPENCL_CHECK_RETURN(cudaMalloc(&d_thread_weights, sizeof(double) * weights_size * actual_threads));

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
	OPENCL_CHECK_RETURN(cudaMalloc(&d_error, sizeof(double)));

	OPENCL_CHECK_RETURN(cudaMemcpy(d_error, &error, sizeof(double), cudaMemcpyHostToDevice));
	compute_error_kernel<<<blocks, threads_per_block>>>(examples, train_settings.regularization_term,
			activations_size, layer_neurons.size(), d_layer_neurons, d_weights, examples, d_x, d_y,
			d_error);
	OPENCL_CHECK_RETURN(cudaGetLastError());
	OPENCL_CHECK_RETURN(cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost));

	int it;
	for(it = 1; it <= train_settings.iterations; ++it) {
		if (error <= train_settings.target_error) {
			break;
		}

		train_kernel<<<blocks, threads_per_block>>>(settings, layer_neurons.size(),
				d_layer_neurons, examples, d_x, d_y, d_weights, d_thread_weights);
		OPENCL_CHECK_RETURN(cudaGetLastError());

		int indexes_per_thread = int_division_round_up(weights_size, actual_threads);
		merge_weights_kernel<<<blocks, threads_per_block>>>(weights_size, indexes_per_thread,
				d_thread_weights, d_weights);
		OPENCL_CHECK_RETURN(cudaGetLastError());

		error = 0;
		OPENCL_CHECK_RETURN(cudaMemcpy(d_error, &error, sizeof(double), cudaMemcpyHostToDevice));
		compute_error_kernel<<<blocks, threads_per_block>>>(examples_per_thread, train_settings.regularization_term,
						activations_size, layer_neurons.size(), d_layer_neurons, d_weights, examples, d_x, d_y,
						d_error);
		OPENCL_CHECK_RETURN(cudaGetLastError());
		OPENCL_CHECK_RETURN(cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost));

		//std::cout << "Iteration " << it << ". Error: " << error << std::endl;
	}

	for(int i = 0; i < weights.size(); ++i) {
		int dimensions = weights[i].get_columns() * weights[i].get_rows();
		OPENCL_CHECK_RETURN(cudaMemcpy(weights[i].get_internal_data_unsafe(),
				d_weights, sizeof(double) * dimensions, cudaMemcpyDeviceToHost));
		d_weights += dimensions;
	}

	d_weights -= weights_size;

	OPENCL_CHECK_RETURN(cudaFree(d_error));
	OPENCL_CHECK_RETURN(cudaFree(d_thread_weights));
	OPENCL_CHECK_RETURN(cudaFree(d_weights));
	OPENCL_CHECK_RETURN(cudaFree(d_y));
	OPENCL_CHECK_RETURN(cudaFree(d_x));
	OPENCL_CHECK_RETURN(cudaFree(d_layer_neurons));

	return TrainResult(min(it, train_settings.iterations), error);
}

double OpenclNeuralNetworkHelper::compute_error(const std::vector<int>& layer_neurons,
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
	OPENCL_CHECK_RETURN(cudaMalloc(&d_error, sizeof(double)));
	OPENCL_CHECK_RETURN(cudaMemcpy(d_error, &error, sizeof(double), cudaMemcpyHostToDevice));

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
	OPENCL_CHECK_RETURN(cudaGetLastError());
	OPENCL_CHECK_RETURN(cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost));

	OPENCL_CHECK_RETURN(cudaFree(d_error));
	OPENCL_CHECK_RETURN(cudaFree(d_weights));
	OPENCL_CHECK_RETURN(cudaFree(d_y));
	OPENCL_CHECK_RETURN(cudaFree(d_x));
	OPENCL_CHECK_RETURN(cudaFree(d_layer_neurons));

	return error;
}

static const char *getOpenclErrorString(cl_int error)
{
	switch(error) {
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
  }
}


/**
 * Check the return value of the OpenCL runtime API call and exit
 * the application if the call has failed.
 */
static void checkOpenclErrorAux(const char *file, unsigned line, const char *statement, cl_int err)
{
	if (err == CL_SUCCESS) {
		return;
	}

	std::cerr << statement <<" returned " << getOpenclErrorString(err) << "(" << err << ") at "
			<< file << ":" << line << std::endl;
	exit(1);
}
