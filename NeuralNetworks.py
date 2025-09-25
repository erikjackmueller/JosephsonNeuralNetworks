import os
import git
import h5py
import yaml
import random
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from Circuit import Circuit
from Utils import add_component
matplotlib.use('Agg')

class NeuronPopulation():
    """
    Class for simulating a Circuit that represents a neuronal Population in Josim
    """
    def __init__(self, parameters=None):
        self.name = 'neuron'
        self.n_neurons = 100
        self.n_connections = 10
        self.R_weight_exc = 1
        self.t_end = 1000
        self.imax = 1e4
        self.rate = 0.1
        self.step_size = 1e-3
        self.out_list = []
        self.print_v = False
        self.kuramoto_values = None
        self.mean_cv = None
        self.noise = 0
        self.name = 'neural_circuit'


        # workaround to get the netlist path
        self.repo = git.Repo('.', search_parent_directories=True)
        self.root = self.repo.working_tree_dir
        self.net_list_path = os.path.join(self.root, 'netlists')

        self.get_names()


        if parameters != None:
            if not ('data_fname' or 'plot_fname' or 'out_fname') in parameters:
                self.__dict__.update(parameters)
                self.get_names()

        self.neuron_phases = [f'PHASE B2.X0.X{k}' for k in range(self.n_neurons)]
        self.neuron_volages = [f'DEVV B2.X0.X{k}' for k in range(self.n_neurons)]
        self.circuit = Circuit(cir_fname=self.cir_fname, out_fname=self.out_fname, plot_fname=self.plot_fname,
                               data_fname=self.data_fname, from_file=False, fade_out_time=50)

    def get_names(self):
        self.out_fname = f'{self.name}.csv'
        self.plot_fname = self.name
        self.data_fname = self.name
        self.cir_fname = os.path.join(self.net_list_path, self.name) + '.cir'

    def build_circuit(self):
        """
        Function that builds a Neural Network of excitatory neurons given the parameters from the __init__()
        It consists of multiple assembley parts:
         1) an Input part where the input to every neuron is created
         2) a neuron subcircuit part, where the neurons are assembled with self.n_connection connections and that are
            hard coded in the netlist file that represents a single neuron and is called 'exc-neuron'
         3) the creation of the neuron network circuit that loads and connects the individual neurons from the previous
            step
        """
        # create circuit from library of subcircuits
        # define a new circuit from subcircuits and give it a new name

        print(f'Building neural network circuit')
        # build n-splitter subcircuit
        self.circuit.build_subcircuit(subcir_fname='netlists/n_splitter.cir', source_file='netlists/Neuron_circuits.cir',
                                 parameters={'n': self.n_connections})
        ports = []

        # reset input for looping over circuits
        self.circuit.input = []
        for i in tqdm(range(1, self.n_neurons + 1), f'building input for {self.n_neurons} neurons'):
            self.circuit.add_input(function='poisson-spikes', ts=np.arange(0.0, self.t_end, self.step_size),
                                   dt=self.step_size, i_max=self.imax, rate=self.rate)

            # set up port map of current neuron
            # port i is the input of the i-th neuron
            possible_port_array = np.arange(self.n_neurons) + 1
            possible_ports = possible_port_array.tolist()
            possible_ports.remove(i)
            port_choice = random.sample(possible_ports, self.n_connections)
            ports_i = [0, i] + port_choice
            ports.append(ports_i)

        # build neuron subcircuit, ################### it has layers ################ !!!!!!!!!!!!!!!!!!!
        neuron_ports = np.arange(self.n_connections + 2).tolist()
        out_ports = neuron_ports[2:]
        out_ports_splitter_syn = [str(k) + '_syn' for k in out_ports]
        neuron_cir_params = {}
        neuron_param_keys = ['subcircuits', 'subcircuit_fnames', 'sub_ports', 'parameters', 'model_names',
                             'types', 'values', 'idx']
        neuron_soma = ['Neuron-Soma', 'netlists/Neuron_circuits.cir', [0, 1, 'soma-split']] + [None] * 4 + [99]
        neuron_splitter = ['2-Split', 'netlists/n_splitter.cir', [0, 'soma-split'] + out_ports_splitter_syn] + [
            None] * 4 + [99]
        add_component(neuron_cir_params, neuron_soma, neuron_param_keys)
        add_component(neuron_cir_params, neuron_splitter, neuron_param_keys)

        for i in range(self.n_connections):
            neuron_synapse_i = ['R-Synapse-exc', 'netlists/Neuron_circuits.cir',
                                [0, out_ports_splitter_syn[i], out_ports[i]],
                                'RSCALE', None, 'default', f'{self.R_weight_exc}', i + 2]
            add_component(neuron_cir_params, neuron_synapse_i, neuron_param_keys)

        self.circuit.build_subcircuit(name='exc-neuron', ports=neuron_ports,
                                      subcir_fname='netlists/exc_neuron.cir',
                                      cir_type='sub-sub', parameters=neuron_cir_params)

        # build exc neuron network
        subcircuits = ['exc-neuron'] * self.n_neurons
        subcircuit_fnames = ['netlists/exc_neuron.cir'] * self.n_neurons
        input_names = [f'IIN{k}' for k in range(self.n_neurons)]

        # define output with optional and phase/voltage parts (voltage is needed to compute FFT of v)
        self.circuit_output_list = self.out_list + self.neuron_phases
        if self.print_v:
            self.circuit_output_list += self.neuron_volages

        self.circuit.build_circuit(subcircuits, subcircuit_fnames, port_list=ports, input_name=input_names,
                                   step_size=self.step_size,
                                   repeat_circuits=False, output_mode='spike-table',
                                   outputs=self.circuit_output_list,
                                   noise=self.noise)

    def run(self, verbose=True, optimize=False, save_log=True):
        """
        Wrapper for Circuit.run() functions
        :param verbose: Bool, default=True
                        specifies if extra printouts are done
        :param optimize: Bool, default=False
                        specifies if paralellisation option is applied to Josim-cli call
        """
        self.circuit.run_simulation(verbose=verbose, optimize=optimize)
        if save_log:
            self.save_log()

    def save_log(self):
        """
        Function that saves the self.__dict__ into a yaml file,
        necessary to repeat experiments
        """
        log_dict = self.__dict__.copy()
        del log_dict['circuit']
        del log_dict['repo']
        print(f'saved log to: {self.data_fname}_log.yaml')
        with open(f'{self.data_fname}_log.yaml', 'w') as file:
            yaml.dump(log_dict, file)


    def load_results(self):
        """
        Wrapper for load_verification function in Circuit
        """
        self.circuit.load_verification(verbose=True, save=True)

    def plot(self, standard=True, raster=True, rate=True, kuramoto=True, save_fig=True, subplot_idxs=[0],
             kuramoto_param=None, two_dim_sweep=False, sweep_data_fname=None, plot_idxs=None, fft=False, fft_param=None,
             time_frame=None, title=True, legend=True, y_labels=None, scaling=None, custom_labels=None, grid=False):
        """
        Wrapper for all-in-one plot functions from Circuit and verification loading
        :param standard:
        :param raster:
        :param rate:
        :param kuramoto:
        :param save_fig:
        :param subplot_idxs:
        :param kuramoto_param:
        :param two_dim_sweep:
        :param sweep_data_fname:
        :param plot_idxs:
        :param fft:
        :param fft_param:
        :param time_frame:
        :param title:
        :param legend:
        :param y_labels:
        :param scaling:
        :param custom_labels:
        :param grid:
        """
        if not os.path.exists(self.out_fname):
            self.out_fname = self.data_fname

        if self.circuit.plot_latex:
            plt.rcParams.update({"text.usetex": True})

        # this part may actually be useless
        phase_idxs = []
        for k in self.out_list:
            if k.lower()[0] == 'p':
                phase_idxs.append(k)

        voltage_idxs = []
        for k in self.out_list:
            if k.lower()[0] == 'v':
                voltage_idxs.append(k)
        if plot_idxs != None:
            standard_plot_idxs = plot_idxs
        else:
            standard_plot_idxs = []
        if standard and (len(self.out_list + standard_plot_idxs) > 0):
            self.circuit.plot_output(plot_idxs=standard_plot_idxs, subplot_idxs=subplot_idxs, save_fig=save_fig,
                                     legend=legend, y_labels=y_labels, scaling=scaling, grid=grid)
            plt.close()
        if raster:
            plot_idxs_raster = np.arange(len(self.out_list), len(self.out_list) + len(self.neuron_phases)).tolist()
            self.circuit.plot_output(mode='raster', save_fig=save_fig, plot_idxs=plot_idxs_raster, title=title,
                                     time_frame=time_frame)
            plt.close()
        if rate:
            mean_cv = self.circuit.get_rate(calc_isi=True, save_fig=save_fig, plot=True, time_frame=time_frame,
                                            show_cv=title, grid=grid)
            self.mean_cv = mean_cv
            print(f'mean coeff of variance is {mean_cv:.5f}')
            plt.close()
        if kuramoto:
            self.kuramoto_param = {'plot': True, 'undersample': None, 'time_frame': time_frame, 'avg': None,
                                   'parameter_idxs': None, 'print_avg': True, 'grid': False, 'title': title}

            if kuramoto_param != None:
                self.kuramoto_param.update(kuramoto_param)

            r, phi = self.circuit.get_kuramoto(plot=self.kuramoto_param['plot'],
                                               save_fig=save_fig,
                                               undersample=self.kuramoto_param['undersample'],
                                               avg=self.kuramoto_param['avg'],
                                               time_frame=self.kuramoto_param['time_frame'],
                                               parameter_idxs=self.kuramoto_param['parameter_idxs'],
                                               grid=self.kuramoto_param['grid'],
                                               title=self.kuramoto_param['title'])
            self.kuramoto_values = [r, phi]
            if self.kuramoto_param['print_avg']:
                print(f'mean kuramoto params: r = {r.mean():.5f}, phi = {phi.mean():.5f}')

        if fft:
            plot_idxs_fft = np.arange(len(self.out_list) + len(self.neuron_phases),
                                      len(self.out_list) + len(self.neuron_phases) + len(self.neuron_volages)).tolist()
            self.fft_param = {'idxs': plot_idxs_fft, 'mean': True}
            if fft_param != None:
                self.fft_param.update(fft_param)
            if not 'stft' in self.fft_param:
                self.circuit.fft(idxs=self.fft_param['idxs'], mean=self.fft_param['mean'], stft=False)
                self.circuit.fft(idxs=self.fft_param['idxs'], mean=self.fft_param['mean'], stft=True)
            else:
                self.circuit.fft(idxs=self.fft_param['idxs'], mean=self.fft_param['mean'], stft=self.fft_param['stft'])
        if two_dim_sweep:

            # This currently plots both the kuramoto order parameter and the coefficient of variance as indicators
            # for synchronisation in the system (high r / kuramoto or low cv /  coeff of variance indicates
            # synchronisation in the system of oscillators

            if sweep_data_fname is None:
                sweep_data_fname = self.data_fname + 'parameter_sweep'
            if os.path.exists(sweep_data_fname + '.hdf5'):
                with h5py.File(sweep_data_fname + '.hdf5', 'r') as h5file:
                    results = np.array(h5file['result'])
                    values = np.array(h5file['values'])
                    parameters = [h5file["parameters"][k].astype(str) for k in range(len(values))]
                cvs = results[:, 1]
                rs = results[:, 0]

                x_shape = len(values[0])
                y_shape = len(values[1])
                r_plot = rs.reshape(x_shape, y_shape).T
                cv_plot = cvs.reshape(x_shape, y_shape).T

                if custom_labels != None:
                    parameters = custom_labels

                fig = plt.figure(figsize=(10, 4.25))

                ax = fig.add_subplot(1, 2, 1)
                X, Y = np.meshgrid(values[0], values[1])
                z_min, z_max = cv_plot.min(), cv_plot.max()
                # z_min, z_max = 1.0, 10.0
                c = ax.pcolormesh(X, Y, cv_plot, cmap='viridis', vmin=z_min, vmax=z_max)
                cb = fig.colorbar(c, ax=ax)
                cb.ax.tick_params(labelsize=self.circuit.plot_label_size)
                if title:
                    ax.set_title('mean coefficient of variance', fontsize=self.circuit.plot_title_size)
                ax.set_xlabel(parameters[0], fontsize=self.circuit.plot_label_size)
                ax.set_ylabel(parameters[1], fontsize=self.circuit.plot_label_size)
                ax.tick_params(labelsize=self.circuit.plot_label_size)

                ax = fig.add_subplot(1, 2, 2)
                z_min, z_max = r_plot.min(), r_plot.max()
                # z_min, z_max = 0.75, 1.0
                c = ax.pcolormesh(X, Y, r_plot, cmap='viridis', vmin=z_min, vmax=z_max)
                cb = fig.colorbar(c, ax=ax)
                cb.ax.tick_params(labelsize=self.circuit.plot_label_size)
                if title:
                    ax.set_title('mean kuramoto order parameter', fontsize=self.circuit.plot_title_size)
                ax.set_xlabel(parameters[0], fontsize=self.circuit.plot_label_size)
                ax.set_ylabel(parameters[1], fontsize=self.circuit.plot_label_size)
                ax.tick_params(labelsize=self.circuit.plot_label_size)
                if save_fig:
                    plt.tight_layout()
                    plt.savefig(self.plot_fname + "_2D_sweep.png")
                    print(f'saved 2D sweep plot to {self.plot_fname}_2D_sweep.png')
                    plt.close()
                else:
                    plt.show()
            else:
                raise FileNotFoundError(f'File {sweep_data_fname}.hdf5 not found!')

    def clean(self, exclude='out_fname'):
        """
        wrapper for Circuit.clear_output()
        """
        self.circuit.clear_output(exclude=exclude)

    def parameter_sweep(self, parameters, values, plot=True, kuramoto=True, cv=True, time_frame=None, start_over=False):
        """
        Performs a sweep over a number of parameters, any number of parameter with all interactions is possible
        The result is stored in the location of self.data_fname as hdf5 along with the logs
        :param parameters: list of lists
                            Parameter names in parameters dict for -> NeuronPopulation(parameters)
        :param values: list of lists
                        Values for the parameters in parameters dict for -> NeuronPopulation(parameters)
        :param plot: Bool, default: True
                    specifies if plots ar generated during sweep
        :param kuramoto: Bool, default: True
                        specifies if kuramoto order parameter is calculated
        :param cv: Bool, default: True
                    specifies if coefficient of variance is calculated
        """

        if type(parameters) != list:
            parameters = [parameters]

        if type(values[0]) == list:
            value_combinations = list(itertools.product(*values))
        else:
            value_combinations = values
            values = [[k] for k in values]

        original_name = self.name
        original_out_fname = self.out_fname
        original_plot_fname = self.plot_fname
        original_data_fname = self.data_fname
        original_cir_fname = self.cir_fname

        if 'time_frame' in parameters:
            time_frame = parameters['time_frame']

        index_start = 0
        if os.path.exists(original_data_fname + "_parameter_sweep.hdf5") and not start_over:
            print(f'reading existing file {original_data_fname}_parameter_sweep.hdf5')
            with h5py.File(original_data_fname + "_parameter_sweep.hdf5", "r") as h5file:
                result = np.array(h5file['result'])
                index_start = np.array(h5file['idx']) + 1
            value_combinations = value_combinations[index_start:]
            print(f'found existing results up to {index_start}')
        else:
            result = np.zeros((len(value_combinations), 2))



        for i, values_i in enumerate(value_combinations):

            # update index if starting with partial sweep data
            idx_string = f'_{i + index_start}'
            end_idx = len(value_combinations) + index_start
            print(f'########################################################################################\n'
                  f'# simulation {i + index_start + 1}/ {end_idx} \n'
                  f'########################################################################################')

            simulation_parameters = {}
            for j in range(len(value_combinations[i])):
                value_ij = values_i[j]
                simulation_parameters[parameters[j]] = value_ij
                print(f'Setting {parameters[j]} = {value_ij}')


            simulation_parameters['name'] = original_name + idx_string
            simulation_parameters['plot_fname'] = original_plot_fname + idx_string
            simulation_parameters['data_fname'] = original_data_fname + idx_string
            simulation_parameters['out_fname'] = original_out_fname + idx_string + '.csv'
            simulation_parameters['cir_fname'] = original_cir_fname + idx_string

            # init again so that a new self.circuit is initialised too
            self.__init__(simulation_parameters)
            # check if files for run exist already and read results from them
            if os.path.exists(self.circuit.data_fname + '_log.yaml'):
                log_dict = yaml.safe_load(Path(self.circuit.data_fname + '_log.yaml').read_text())
                start_time = 0
                if 't_start' in log_dict:
                    start_time = log_dict['t_start']
                end_time = log_dict['t_end'] + self.circuit.fade_out_time
                step_size = log_dict['step_size']
                self.circuit.time = np.arange(start_time, end_time, step_size) * 1e-12

            if not os.path.exists(self.circuit.data_fname + '_verification.csv') or not \
                    os.path.exists(self.circuit.out_fname):
                self.build_circuit()
                self.run(save_log=True, optimize=True)
            self.load_results()

            if plot:
                self.plot(save_fig=True, time_frame=time_frame)

            if kuramoto:
                if self.kuramoto_values !=None:
                    r = np.mean(self.kuramoto_values[0])
                else:
                    r = self.circuit.get_kuramoto(avg=True, time_frame=time_frame, undersample=None)[0]
                result[i, 0] = r
            if cv:
                if self.mean_cv != None:
                    cv = self.mean_cv
                else:
                    cv = self.circuit.get_rate(calc_isi=True, time_frame=time_frame)
                result[i, 1] = cv

            self.clean()

            # save data in every iteration to get inbetween data
            print(f'saving sweep data to {original_data_fname}_parameter_sweep.hdf5')
            with h5py.File(original_data_fname + "_parameter_sweep.hdf5", "w") as h5file:
                h5file.create_dataset('parameters', data=np.array(parameters, dtype='S50'))
                h5file.create_dataset('values', data=np.array(values))
                h5file.create_dataset('result', data=np.array(result))
                h5file.create_dataset('idx', data=i)