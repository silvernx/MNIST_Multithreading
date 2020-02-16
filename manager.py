import network

networks = []
errors = []

def build_networks(layers, activations, d_activations, cost, d_cost, num_nets,
        random_limit):
    for i in range(num_nets):
        networks.append(network.FeedForwardNetwork(layers, activations, d_activations, cost, d_cost, random_limit))
        errors.append(0)

def train_nets(inputs, outputs, training_rate, epochs, batch_size, outer_min,
        random_limit, layers, activations, d_activations, cost, d_cost, num_nets):
    minimum = 100
    minnet = 3
    while minimum > outer_min:
        build_networks(layers, activations, d_activations, cost, d_cost, num_nets, random_limit)
        for network in networks:
            print("barrier1")
            output = network.train(inputs, outputs, training_rate, epochs,
                    batch_size, True)
            if output < minimum:
                minimum = output
                minnet = network

        print("Couldn't find anything")
        print(minimum, random_limit)
        random_limit *= 10
        networks.clear()

    return minnet
