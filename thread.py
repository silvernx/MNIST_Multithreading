import threading

class worker_thread(threading.Thread):
    def __init__(self, threadID, inputs, outputs, network_copy, training_rate, deltabiases, deltaweights, barrier, batch_size, num_threads,
            main_thread_event, worker_thread_event, mutex1, mutex2):
        super().__init__()
        self.threadID = threadID
        self.inputs = inputs
        self.outputs = outputs
        self.network_copy = network_copy
        self.training_rate = training_rate
        self.deltabiases = deltabiases
        self.deltaweights = deltaweights
        self.batch_number = 0
        self.num_threads = num_threads
        self.barrier = barrier
        self.main_thread_event = main_thread_event
        self.worker_thread_event = worker_thread_event
        self.batch_size = batch_size
        self.mutex1 = mutex1
        self.mutex2 = mutex2
        self.done = False

    def run(self):
        while not self.done:
            self.update()
            self.barrier.wait()
            if self.threadID == 0:
                self.deltabiases = [0 for bias in self.deltabiases]
                self.deltaweights = [0 for weight in self.deltaweights]
            self.barrier.wait()
            self.train()
            self.update_main_thread()
            self.wait_for_other_threads()
            self.batch_number += 1
            self.batch_number %= len(self.inputs) / self.batch_size

    def update(self):
        weight_index = 0
        bias_index = 0
        for layer in self.network_copy.layers:
            for neuron in layer:
                neuron.bias += self.deltabiases[bias_index]
                neuron.deltabias = 0
                neuron.delta = 0
                bias_index += 1
                for _, weight in neuron.children:
                    weight.weight += self.deltaweights[weight_index]
                    weight.deltaweight = 0
                    weight_index += 1

    def train(self):
        start = int(self.batch_number * len(self.inputs) / self.batch_size + self.batch_size / self.num_threads * self.threadID)
        for i in range(int(self.batch_size / self.num_threads)):
            self.network_copy.prop_to_and_fro(self.inputs[i + start], self.outputs[i + start], self.training_rate)

    def wait_for_other_threads(self):
        self.barrier.wait()
        self.main_thread_event.set()
        if self.threadID == 0:
            for i in range(len(self.deltaweights)):
                self.deltaweights[i] = self.deltaweights[i] / self.num_threads
            for i in range(len(self.deltabiases)):
                self.deltabiases[i] = self.deltabiases[i] / self.num_threads
        self.barrier.wait()

    def update_main_thread(self):
        index1 = 0
        index2 = 0
        for layer in self.network_copy.layers:
            for neuron in layer:
                self.mutex1.acquire()
                self.deltabiases[index1] += neuron.deltabias
                self.mutex1.release()
                index1 += 1
                for _,weight in neuron.children:
                    self.mutex2.acquire()
                    self.deltaweights[index2] += weight.deltaweight
                    self.mutex2.release()
                    index2 += 1
