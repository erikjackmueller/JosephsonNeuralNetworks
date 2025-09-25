from NeuralNetworks import NeuronPopulation

# set up new file structure for test
# change folder depending on device, optimally execute from hpc
time_start = 0.2
time_frame = [int(1e6*time_start)]
parameters = {'name': 'async1', 'n_connections': 5, 'n_neurons': 100, 'imax': 1e4, 'rate': 0.4, 'print_v': True,
              'noise': 4.2}

nn = NeuronPopulation(parameters=parameters)
nn.build_circuit()
nn.run(save_log=True)
nn.load_results()

# more customization from Circuit class plot
nn.circuit.plot_label_size = 20
nn.circuit.plot_latex = True
nn.circuit.plot_output(plot_idxs=[2, 3, 4, 5, 6, 102, 103, 104, 105, 106], subplot_idxs=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                       y_labels=['Phase', 'v in mV'], scaling=[1, 1, 1, 1, 1, 1e3, 1e3, 1e3, 1e3, 1e3], save_fig=True,
                       legend=False, latex=True, time_frame=time_frame, grid=False)
nn.circuit.plot_label_size = 20
nn.plot(time_frame=time_frame, raster=True, rate=True, grid=False, title=True)
nn.clean(exclude=[])

#####
# 2
#####

parameters = {'name': 'async2', 'n_connections': 15, 'n_neurons': 100, 'imax': 1e4, 'rate': 0.5, 'print_v': True,
              'noise': 4.2}
nn = NeuronPopulation(parameters=parameters)
# nn.build_circuit()
# nn.run(save_log=True)
nn.load_results()
# more customization from Circuit class plot
nn.circuit.plot_label_size = 20
nn.circuit.plot_latex = True
nn.circuit.plot_output(plot_idxs=[2, 3, 4, 5, 6, 102, 103, 104, 105, 106], subplot_idxs=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                       y_labels=['Phase', 'v in mV'], scaling=[1, 1, 1, 1, 1, 1e3, 1e3, 1e3, 1e3, 1e3], save_fig=True,
                       legend=False, latex=True, time_frame=time_frame, grid=False)
nn.circuit.plot_label_size = 20
nn.plot(time_frame=time_frame, raster=True, rate=True, grid=False, title=True)
nn.clean(exclude=[])

#####
# 3
#####

parameters = {'name': 'sync1', 'n_connections': 25, 'n_neurons': 100, 'imax': 1e4, 'rate': 0.7, 'print_v': True,
              'noise': 4.2}
nn = NeuronPopulation(parameters=parameters)
# nn.build_circuit()
# nn.run(save_log=True)
nn.load_results()
# more customization from Circuit class plot
nn.circuit.plot_label_size = 30
nn.circuit.plot_latex = True
nn.circuit.plot_output(plot_idxs=[2, 3, 4, 5, 6, 102, 103, 104, 105, 106], subplot_idxs=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                       y_labels=['Phase', 'v in mV'], scaling=[1, 1, 1, 1, 1, 1e3, 1e3, 1e3, 1e3, 1e3], save_fig=True,
                       legend=False, latex=True, time_frame=time_frame, grid=False)
nn.circuit.plot_label_size = 20
nn.plot(time_frame=time_frame, raster=True, rate=True, grid=False, title=True)
nn.clean(exclude=[])

#####
# 4
#####

parameters = {'name': 'special', 'n_connections': 45, 'n_neurons': 100, 'imax': 1e4, 'rate': 0.8, 'print_v': True,
              'noise': 4.2}
nn = NeuronPopulation(parameters=parameters)
# nn.build_circuit()
# nn.run(save_log=True)
nn.load_results()
# more customization from Circuit class plot
nn.circuit.plot_label_size = 20
nn.circuit.plot_latex = True
nn.circuit.plot_output(plot_idxs=[2, 3, 4, 5, 6, 102, 103, 104, 105, 106], subplot_idxs=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                       y_labels=['Phase', 'v in mV'], scaling=[1, 1, 1, 1, 1, 1e3, 1e3, 1e3, 1e3, 1e3], save_fig=True,
                       legend=False, latex=True, time_frame=time_frame, grid=False)
nn.circuit.plot_label_size = 20
nn.plot(time_frame=time_frame, raster=True, rate=True, grid=False, title=True)
nn.clean(exclude=[])
