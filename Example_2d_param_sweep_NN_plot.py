from NeuralNetworks import NeuronPopulation

file = ''  # insert filename of 2D sweep data

nn = NeuronPopulation({'name': 'nn_n_con_r_exc_sweep'})
nn.circuit.plot_latex = True
nn.circuit.plot_label_size = 15
nn.plot(standard=False, raster=False, rate=False, kuramoto=False, save_fig=True, two_dim_sweep=True,
        sweep_data_fname=file, title=False, custom_labels=['Poisson rate', 'n_c'])
