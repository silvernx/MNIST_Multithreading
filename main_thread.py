from network import *
import manager
import thread
import threading
import numpy as np

class main_thread(threading.Thread):
    def __init__(self, architecture, f_activations, d_f_activations, f_cost, d_f_cost, random_limit, num_threads, inputs, outputs, training_rate, batch_size, epochs, network):
        super().__init__()
        self.barrier = threading.Barrier(num_threads)
        self.main_thread_event = threading.Event()
        self.worker_thread_event = threading.Event()
        self.worker_threads = []
        self.deltabiases = []
        self.deltaweights = []
        self.batch_size = batch_size
        self.inputs = inputs
        self.outputs = outputs
        self.num_threads = num_threads
        self.epochs = epochs
        self.training_rate = training_rate
        self.networks = [network]
        mutex1 = threading.Lock()
        mutex2 = threading.Lock()
        for layer in self.networks[0].layers:
            for neuron in layer:
                self.deltabiases.append(0)
                for child in neuron.children:
                    self.deltaweights.append(0)
        for i in range(num_threads - 1):
            self.networks.append(self.networks[0].copy())
        for i in range(num_threads):
            self.worker_threads.append(thread.worker_thread(i, inputs, outputs, self.networks[i], self.training_rate, self.deltabiases, self.deltaweights, self.barrier, self.batch_size, self.num_threads, self.main_thread_event, self.worker_thread_event, mutex1, mutex2))

    def run(self):
        barrier = threading.Barrier(self.num_threads)
        for i in range(self.num_threads):
            self.worker_threads[i].start()
        for i in range(self.epochs):
            data = list(zip(self.inputs, self.outputs))
            np.random.shuffle(data)
            self.inputs, self.outputs = zip(*data)
            print('Epoch #' + str(i))
            if (i+1) % 10 == 0:
                self.training_rate *= 5
            for j in range(len(self.inputs) // self.batch_size):
                self.main_thread_event.clear()
                self.main_thread_event.wait()
                self.deltabiases = [bias / self.num_threads for bias in self.deltabiases]
                self.deltaweights = [weight / self.num_threads for weight in self.deltaweights]
                self.worker_thread_event.set()
        for thread in self.worker_threads:
            thread.done = True
        for thread in self.worker_threads:
            thread.join
