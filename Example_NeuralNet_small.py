from NeuralNetworks import NeuronPopulation


time_start = 0.2
time_frame = [int(1e6*time_start)]

parameters = {'n_connections': 3, 'n_neurons': 10, 'imax': 1e6, 'rate': 0.2, 'print_v': False, 'noise':4.2} #, 't_end': 5000}

nn = NeuronPopulation(parameters=parameters)
nn.build_circuit()
nn.run(save_log=True)
nn.load_results()
nn.plot(plot_idxs=[2, 3, 4, 12, 13, 14], subplot_idxs=[0, 0, 0, 1, 1, 1], time_frame=time_frame, fft=False)
nn.clean(exclude=[])

