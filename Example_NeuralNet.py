from NeuralNetworks import NeuronPopulation
import os

# set up new file structure for test
# change folder depending on device, optimally execute from hpc

folder = 'D:\\NN'
name = 'async'
plot_files = os.path.join(folder, 'plots')
plot_fname = os.path.join(plot_files, name)
data_files = os.path.join(folder, 'data')
data_fname = os.path.join(data_files, name)
parameters = {'n_connections': 20, 'name': name, 'plot_fname': plot_fname, 'data_fname': data_fname, 'rate':0.2,
              'time_frame':[200000]}

nn = NeuronPopulation(parameters=parameters)
nn.build_circuit()
nn.run(save_log=True)
nn.load_results()
nn.plot(save_fig=True)
nn.clean(exclude=[])
