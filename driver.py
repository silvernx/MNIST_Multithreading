from network import *
from main_thread import *
import pickle
import gzip
import numpy as np
import manager

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere."""

    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def main():
    training_data, validation_data, test_data = load_data_wrapper()
    training_inputs, training_results = zip(*training_data)
    Validation_inputs, Validation_results = zip(*validation_data)
    print(len(Validation_inputs))
    training_inputs_small=training_inputs[0:1000]
    training_results_small=training_results[0:1000]
    random_limit = 20
    batch_size = 10
    outer_min = 1
    training_rate = 1
    architecture = [784, 20, 15, 10]
    num_nets = 1
    final = manager.train_nets(training_inputs_small, training_results_small, training_rate, 1, batch_size, outer_min, random_limit, architecture,
            [sigmoid] * 3, [d_sigmoid] * 3, squared_error, d_squared_error, num_nets)
    print("Start Deep Training")
    training_inputs_medium=training_inputs[0:10000]
    training_results_medium=training_results[0:10000]
    training_rate = 1
    #inputs = [[0,0],[1,0],[0,1],[1,1]]
    #outputs = [[0],[1],[1],[0]]
    f_activations = [sigmoid] * 3
    d_f_activations = [d_sigmoid] * 3
    f_cost = squared_error
    d_f_cost = d_squared_error
    batch_size = 50
    epochs = 5
    num_threads = 10
    master_thread = main_thread(architecture, f_activations, d_f_activations, f_cost, d_f_cost, random_limit, num_threads, training_inputs_small, training_results_small, training_rate, batch_size, epochs, final)
    master_thread.start()
    master_thread.join()

    print("Start Validation Tests")
    Validation_cnt = 0
    for i in range(1000):
        test_input=Validation_inputs[i]
        test_output=Validation_results[i]
        ret = final.prop(test_input)
        network.final.clear()
        max = 0
        for j in range(10):
            if ret[j] > max:
                max=ret[j]
                opt=j
        if opt == test_output:
            Validation_cnt +=1
        print(opt, test_output)
    print("Correct Answer =", Validation_cnt)
    print("Success Rate:", Validation_cnt/1000)


if __name__ == '__main__':
    main()
