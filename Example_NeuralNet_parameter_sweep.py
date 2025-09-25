from NeuralNetworks import NeuronPopulation
import os

def main():
    # set up new file structure for test
    # change folder depending on device, optimally execute from hpc

    folder = 'D:\\NN_param_sweep_new'
    name = 'NN_sweep'
    plot_files = os.path.join(folder, 'plots')
    plot_fname = os.path.join(plot_files, name)
    data_files = os.path.join(folder, 'data')
    data_fname = os.path.join(data_files, name)
    out_fname = data_fname
    parameters = {'name': name, 'plot_fname': plot_fname, 'data_fname': data_fname, 'out_fname': out_fname}

    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    con = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    sweep_parameters = ['rate', 'n_connections']
    sweep_values = [rates, con]

    nn = NeuronPopulation(parameters=parameters)
    nn.parameter_sweep(parameters=sweep_parameters, values=sweep_values, time_frame=[200000], start_over=True)
    # time frame equivalent to 0.2ns with step size of 1e-3ps


if __name__ == '__main__':
    main()