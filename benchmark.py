import timeit

times_per_command = 3

def benchmark(base_command, argument_strings):
    print "Benchmarking", base_command
    return [timeit.timeit('__import__("os").system("{0} {1}")'.format(base_command, str(arg_string)),
        number=times_per_command) for arg_string in argument_strings]

def write_result_file(results, file_name):
    with open("results/" + file_name, "w") as f:
        for result in results:
            f.write("{0} {1}\n".format(str(result[0]), str(result[1])))

def full_benchmark(file_prefix, command, argument_strings, names):
    results = benchmark(command, argument_strings)

    write_result_file(zip(names, results), file_prefix + "_results.txt")

    speedup = [(float(results[0]) / item) for item in results]

    write_result_file(zip(names, speedup), file_prefix + "_speedup.txt")

momentum = 0.9
learning_rate = 1.0
epsilon = 10
#iterations = 100
iterations = 1
target_error = 0
steps_to_simulate = 100000


def generate_arg_string(inner_steps, threads, blocks=1):
    return "-m %f -l %f -e %f -t %d -b %d -s %d -i %d --error %f" % (momentum,
        learning_rate, epsilon, threads, blocks, inner_steps, iterations, target_error)

def generate_same_problem_arg_string(threads, blocks=1):
    return generate_arg_string((steps_to_simulate + threads - 1) / threads, threads, blocks)

CUDA_BINARY = './neural-networks-cuda/build/bin/runner'
PTHREAD_BINARY = './neural_networks/build/bin/runner'
OPENMP_BINARY = './neural-networks-openmp/build/bin/runner'

"""thread_names = range(1, 16)
thread_args = [generate_arg_string(100, i) for i in thread_names]

full_benchmark("nn_pthreads", PTHREAD_BINARY, thread_args, thread_names)
full_benchmark("nn_openmp", OPENMP_BINARY, thread_args, thread_names)

thread_names = range(1, 1001, 50)
thread_args = [generate_arg_string(100, i, blocks=10) for i in thread_names]

full_benchmark("nn_cuda", CUDA_BINARY, thread_args, thread_names)"""

thread_names = range(1, 16)
thread_args = [generate_same_problem_arg_string(i) for i in thread_names]

full_benchmark("nn_pthreads", PTHREAD_BINARY, thread_args, thread_names)
full_benchmark("nn_openmp", OPENMP_BINARY, thread_args, thread_names)

thread_names = range(1, 1001, 50)
thread_args = [generate_same_problem_arg_string(i, blocks=10) for i in thread_names]

full_benchmark("nn_cuda", CUDA_BINARY, thread_args, thread_names)
