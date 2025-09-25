from NeuralNetworks import NeuronPopulation

# set up new file structure for test
# change folder depending on device, optimally execute from hpc
time_start = 0.2
time_frame = [int(1e6*time_start)]
# parameters = {'n_connections': 10, 'n_neurons': 100, 'imax': 1e4, 'rate': 0.5, 'print_v': True, 'name': 'nn_sync'}
parameters = {'n_connections': 5, 'n_neurons': 100, 'imax': 1e4, 'rate': 0.7, 'print_v': True, 'name': 'nn_async'}
# parameters = {'n_connections': 3, 'n_neurons': 10, 'imax': 1e6, 'rate': 0.2, 'print_v': True} #, 't_end': 5000}

nn = NeuronPopulation(parameters=parameters)
nn.circuit.plot_latex = True
nn.circuit.plot_label_size = 15
nn.build_circuit()
nn.run(save_log=True)
nn.load_results()
nn.plot(plot_idxs=[2, 3, 4, 5, 6, 102, 103, 104, 105, 106], subplot_idxs=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        time_frame=time_frame, legend=False, title=False, y_labels=['Phase', 'v (mV)'], scaling=[1]*5 + [1e3]*5)
nn.clean(exclude=[])

