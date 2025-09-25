import warnings
import time
import csv
import copy
import scipy.fft as fft
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from Utils import *

matplotlib.use('TkAgg')


class Circuit():
    """ Class that is used to read, build, simulate and analyze circuits with josim """

    def __init__(self, from_file=True, cir_fname=None, out_fname=None, plot_fname=None,
                 step_size='0.20p', output_step_size=None, output_start=0, fade_out_time=200,
                 gpc_options=None, margin_dict=None, data_fname=None, plot_latex=False, plot_label_size=15):
        self.name = 'circuit'
        self.from_file = from_file

        if from_file:
            if cir_fname is not None:
                self.cir_fname = cir_fname
                if cir_fname[-4:] != '.cir':
                    raise AttributeError('Please specify a cir_fname as .cir file!')

                with open(self.cir_fname, "r") as file:
                    self.original_lines = file.readlines()
            else:
                raise AttributeError('Please specify a cir_fname!')
        else:
            if cir_fname is None:
                cir_fname = 'josim_circuit.cir'
                print(f'Setting circuit name to {cir_fname}')
            else:
                self.cir_fname = cir_fname
        # temporary circuit fname for deleting if build_circuit was used
        self.temp_cir_fname = None

        if plot_fname is None:
            self.plot_fame = self.cir_fname.split()[:-4]
        else:
            self.plot_fname = plot_fname

        if out_fname is None:
            self.out_fname = 'temp.csv'
        else:
            self.out_fname = out_fname

        if data_fname is None:
            self.data_fname = self.out_fname
        else:
            self.data_fname = data_fname

        self.step_size = step_size
        self.output_step_size = output_step_size
        if self.output_step_size is None:
            self.output_step_size = self.step_size
        self.output_start = output_start
        self.fade_out_time = fade_out_time

        # instantiate the input class inside as list
        # new input can then be created outside and added
        # Example:
        # >>> new_input = <circuit>.Input(<input_kwargs>)
        # >>> new_input_name = new_input.name
        # >>> <circuit>.input.append(new_input)
        # >>> new_input_in_circuit = <circuit>.input[1]
        self.input = []
        self.temp_files = []

        self.verification = None
        self.output_keys = []
        self.time = None
        self.plot_latex = plot_latex
        if self.plot_latex:
            plt.rcParams.update({"text.usetex": True})
        self.plot_label_size = plot_label_size
        self.plot_title_size = int(1.5 * self.plot_label_size)

    def run_simulation(self, temp_cir_fname=None, temp_out_fname=None, spec_needed=False, index_list=None, save=False,
                       verbose=False, optimize=False):
        """
        gets the time values and saves as csv/hdf5
        for each important junction those values need to be exported and carried over to a verification file
        in Delport_2023 they only sampled the number 2pi phase transitions after the same time step which I find to be
        less precise
        :param temp_cir_fname:
        :param temp_out_fname:
        :param spec_needed:
        :param index_list:
        :return: step_times:
        """
        if temp_cir_fname is None:
            temp_cir_fname = self.cir_fname
        if temp_out_fname is None:
            temp_out_fname = self.out_fname

        if temp_cir_fname[0] != 'C' and temp_cir_fname[0] != 'D' and temp_cir_fname[0:5] != '/home':
            run_cir_fname = "./" + temp_cir_fname
        else:
            run_cir_fname = temp_cir_fname

        if temp_out_fname[0] != 'C' and temp_out_fname[0] != 'D' and temp_out_fname[0:5] != '/home':
            run_output_fname = "./" + temp_out_fname
        else:
            run_output_fname = temp_out_fname

        if not optimize:
            exec_string = "josim-cli -o " + run_output_fname + " " + run_cir_fname + " -V 1"
        else:
            exec_string = "josim-cli -o " + run_output_fname + " " + run_cir_fname + " -V 1" + " -p"
        t_start = time.time()
        try:
            if not verbose:
                execute_silently(exec_string)
            else:
                os.system(exec_string)
        except:
            SystemError('Please check that josim is correctly installed and verify by calling "josim-cli" in a'
                        ' terminal!')
        t_end = time.time()
        time_needed, time_unit = t_format(t_end - t_start)

        if verbose:
            print(f'Josim simulation time: {time_needed:.3f}' + time_unit)
        if spec_needed:
            # read the output file and record step times
            df = pd.read_csv(temp_out_fname, sep=',')

            # extract time as first row in .csv
            keynames = df.keys()
            time_vals = df['time']

            # extract remaining values and format to array
            key_length = len(keynames)
            values = []
            for i in range(1, key_length):
                if 'p(' in keynames[i].lower():
                    values.append(df[keynames[i]])
            # values = np.array([np.array(df[keynames[i + 1]]) for i in range(key_length)])
            # extract phase value in this case, get relevant indexes for this step beforehand
            step_times = []
            if index_list == None:
                index_list = np.arange(len(values))
            if verbose:
                for i in tqdm(index_list, f'creating verification for {len(values)} values'):
                    # steps = get_steps(values[i])
                    step_mask = get_Steps_mask(values[i], noise_tolerance=0)
                    steps = np.where(step_mask == 1)[0]
                    step_times.append(time_vals[steps].to_numpy())
            else:
                for i in index_list:
                    # steps = get_steps(values[i])
                    step_mask = get_Steps_mask(values[i], noise_tolerance=0)
                    steps = np.where(step_mask == 1)[0]
                    step_times.append(time_vals[steps].to_numpy())
            self.verification = step_times

            if save:
                print(f'save verification to {self.out_fname[:-4]}_verification.hdf5')
                with h5py.File(self.out_fname[:-4] + "_verification.hdf5", "w") as h5file:
                    h5file.create_dataset('verification', data=step_times)
                self.temp_files.append(f'{self.out_fname[:-4]}_verification.hdf5')

            return step_times

    def josim_native_plot(self):
        """
        using josims native python file josim-plot.py to plot the results
        consult josim docu for information on additional kwargs and alterations
        """
        os.system("python josim-plot.py ./" + self.out_fname + " -o " + self.plot_fname +
                  "_josim_native.png -t stacked -c light")

    def get_rate(self, calc_isi=False, size=(6, 4), idxs=None, save_fig=False, plot=False, time_frame=None,
                 plot_isi=False, dpi=600, show_cv=True, grid=False, no_frame=True):

        plt.rcParams['xtick.labelsize'] = self.plot_label_size
        plt.rcParams['ytick.labelsize'] = self.plot_label_size

        if idxs == None:
            data = self.verification
        else:
            data = self.verification[idxs]

        if type(data[0]) == float:
            data = [data]
            n_neurons = 1
        else:
            n_neurons = len(data)

        # create time grid and map a 1 to this grid, for every spike that occurred in this time frame
        t = self.time
        dt = np.diff(self.time)[0]
        t_size = t.shape[0]
        r = np.zeros((n_neurons, t_size))
        isi = np.zeros((n_neurons, t_size - 1))
        coeff_of_var = np.zeros_like(t)
        for i, neuron_rate in enumerate(data):
            try:
                neuron_rate = np.array(neuron_rate)
                spike_idxs = round_to_1dgrid(neuron_rate, t, idx=True)[1]
            except:
                raise ValueError(f'irregular neuron rate found = {neuron_rate}?')
            if t_size in spike_idxs:
                spike_idxs = spike_idxs[:-1]
            r[i, spike_idxs] = 1
            if calc_isi and neuron_rate.shape[0] > 0:
                isi_vals = np.diff(neuron_rate)
                last_spike_interval = np.array(t)[-1] - neuron_rate[-1]
                # isi[i] = np.mean(isi_vals)
                isi[i, spike_idxs[1:]] = isi_vals
                isi[i, -1] = last_spike_interval
                isi[i, :] = stretch_spikes(isi[i, :])
        r /= n_neurons # scale by number of neurons
        isi_exists = isi.any()

        if isi_exists:
            isi_mean = np.mean(isi, axis=0)
            isi_std = np.std(isi, axis=0)
            coeff_of_var = isi_std / isi_mean

        if time_frame != None:
            if len(time_frame) == 1:
                coeff_of_var = coeff_of_var[time_frame[0]:]
                t = t[time_frame[0]:]
                r = r[:, time_frame[0]:]
                if isi_exists:
                    isi_mean = isi_mean[time_frame[0]:]
            else:
                coeff_of_var = coeff_of_var[time_frame[0], time_frame[1]]
                t = t[time_frame[0], time_frame[1]]
                r = r[:, time_frame[0], time_frame[1]]
                if isi_exists:
                    isi_mean = isi_mean[time_frame[0]:]
        mean_cv = np.nanmean(coeff_of_var)
        if plot:
            t_labels = np.array(t) * 1e9  # convert to ns
            r_sum = np.sum(r, axis=0) * (1 / dt) * 1e-9  # convert to GHz
            sigma = int(t.shape[0] / 200)
            r_smooth = gaussian_filter1d(r_sum, sigma=sigma)

            n_subplots = 1
            fig, ax = plt.subplots(n_subplots, 1, figsize=size)
            line = ax.plot(t_labels, r_smooth)
            ax.set_xlabel('time in ns', fontsize=self.plot_label_size)
            # ax.set_ylabel('firing rate in GHz', fontsize=self.plot_label_size)
            ax.set_ylabel('firing rate (a.u.)', fontsize=self.plot_label_size)
            ax.set_xlim(t_labels[0], t_labels[-1])

            if no_frame:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            # plt.tight_layout()
            if grid:
                plt.grid()
            if isi_exists:
                if show_cv:
                    # ax.text(0.1, 0.9, f'CV: {mean_cv:.3f}', fontsize=20, transform=ax.transAxes, va='top')
                    ax.set_title(f'CV: {mean_cv:.3f}')
            if save_fig:
                plt.savefig(f'{self.plot_fname}_rate.png', dpi=dpi, bbox_inches='tight')
                print(f'saved rate plot to {self.plot_fname}_rate.png')
                plt.close()
            else:
                plt.show()

        if plot_isi and calc_isi:
            t_labels = np.array(t) * 1e9  # convert to ns
            n_subplots = 1
            fig, ax = plt.subplots(1, 1, figsize=size)
            line = ax.plot(t_labels[:-1], isi_mean * 1e9)
            ax.set_xlabel('time in ns', fontsize=self.plot_label_size)
            # ax.set_ylabel('Inter-Spike-Interval in ns', fontsize=self.plot_label_size)
            ax.set_ylabel('Inter-Spike-Interval (a.u.)', fontsize=self.plot_label_size)
            ax.set_xlim(t_labels[0], t_labels[-2])

            if no_frame:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            if grid:
                plt.grid()
            plt.tight_layout()

            if save_fig:
                plt.savefig(f'{self.plot_fname}_isi.png', dpi=dpi, bbox_inches='tight')
                print(f'saved rate plot to {self.plot_fname}_isi.png')
                plt.close()
            else:
                plt.show()
        return mean_cv

    def plot_output(self, plot_idxs=None, subplot_idxs=[1], size=(12, 8), compare_fname=None,
                    save_fig=False, custom_labels=None, mode='default', latex=False, grid=False, y_labels=None,
                    scaling=None, fname=None, legend=True, title=True, time_frame=None, no_frame=True):

        """
        Plotting function for josim circuits
        :param plot_idxs: list of int, indices of output parameters that are plotted
        :param subplot_idxs: list of int, index of the subplot, where the parameter should be plotted, default=[1]
        :param size: tuple, size of the plot, default is (12, 8)
        :param compare_fname: string, file name of data that is compared against
        :param save_fig, bool, option to save the figure instead of showing
        """
        if latex or self.plot_latex:
            plt.rcParams.update({"text.usetex": True})


        plt.rcParams['xtick.labelsize'] = self.plot_label_size
        plt.rcParams['ytick.labelsize'] = self.plot_label_size

        if mode == 'default':
            # check if compare option is checked

            if compare_fname == None:
                # standard plot
                data = pd.read_csv(self.out_fname, sep=',')
                keys = data.keys()
                if scaling == None:
                    scaling = [1] * len(subplot_idxs)
                if time_frame != None:
                    new_data = {}
                    for k, key in enumerate(keys):
                        if len(time_frame) == 1:
                            new_data[key] = data[key][time_frame[0]:]
                        else:
                            new_data[key] = np.array(data[key][time_frame[0]:time_frame[1]])
                    data = new_data
                if plot_idxs != None:
                    if len(subplot_idxs) == len(plot_idxs) and len(scaling) == len(plot_idxs):
                        n_subplots = int(np.max(np.array(subplot_idxs))) + 1
                        print(f' plotting {[keys[k] for k in plot_idxs]}')
                        fig, ax = plt.subplots(n_subplots, 1, figsize=size)
                        if n_subplots > 1:
                            for i, idx in enumerate(plot_idxs):
                                if custom_labels == None:
                                    init_label = keys[idx]
                                    label = format_key_name(init_label)
                                else:
                                    label = custom_labels[i]

                                ax[subplot_idxs[i]].plot(np.array(data['time']) * 1e9,
                                                         scaling[i] * np.array(data[keys[idx]]),
                                                         label=label, lw=3)
                                # general formatting of all axes
                            for j in range(n_subplots):
                                ax[j].set_xlabel('t in ns', fontsize=self.plot_label_size)
                                if legend:
                                    ax[j].legend(bbox_to_anchor=(0.97, 1.0), loc=1)
                                if grid:
                                    ax[j].grid()
                                if y_labels != None:
                                    ax[j].set_ylabel(y_labels[j], fontsize=self.plot_label_size)
                                x_lim = (np.array(data['time'] * 1e9)[0], np.array(data['time'] * 1e9)[-1])
                                ax[j].set_xlim(x_lim)
                        else:
                            for i, idx in enumerate(plot_idxs):
                                ax.plot(data['time'] * 1e9, scaling[0] * data[keys[idx]])
                            # general formatting of all axes
                            ax.set_xlabel('t in ns', fontsize=self.plot_label_size)
                            if y_labels != None:
                                ax.set_ylabel(y_labels[0], fontsize=self.plot_label_size)
                            x_lim = (np.array(data['time'] * 1e9)[0], np.array(data['time'] * 1e9)[-1])
                            ax.set_xlim(x_lim)
                    else:
                        raise ValueError('Number of plot_idxs, number of subplot_idxs and number of scaling factors'
                                         ' must be the same!')

                else:
                    n_subplots = int(len(keys) - 1)
                    fig, ax = plt.subplots(n_subplots, 1, figsize=size)
                    if n_subplots > 0:
                        for i, key in enumerate(keys):
                            if i != 0:
                                ax[i - 1].plot(np.array(data['time']) * 1e9, scaling[i] * np.array(data[key]),
                                               label=key)
                                # general formatting of
                        for j in range(n_subplots):
                            ax[j].set_xlabel('t in ns', fontsize=self.plot_label_size)
                            if grid:
                                ax[j].grid()
                            if legend:
                                ax[j].legend(bbox_to_anchor=(0.97, 1.0), loc=1)
                            if y_labels != None:
                                ax[j].set_ylabel(y_labels[j], fontsize=10)
                    else:
                        for i, key in enumerate(plot_idxs):
                            ax.plot(data['time'] * 1e9, scaling[i] * data[key])
                        if y_labels != None:
                            ax.set_ylabel(y_labels[0], fontsize=self.plot_label_size)
                        # general formatting of all axes
                        ax.set_xlabel('t in ns')

            else:
                data = [pd.read_csv(self.out_fname, sep=','), pd.read_csv(compare_fname, sep=',')]
                keys = [data[0].keys(), data[1].keys()]
                n_subplots = int(len(keys[0]) - 1)
                if scaling == None:
                    scaling = [1] * n_subplots
                fig, ax = plt.subplots(n_subplots, 1, figsize=size)
                for k in (0, 1):
                    if n_subplots > 0:
                        for i, key in enumerate(keys[k]):
                            if i != 0:
                                ax[i - 1].plot(np.array(data[k]['time']) * 1e9, scaling[i-1] * np.array(data[k][key]),
                                               label=key)
                                # general formatting of
                        for j in range(n_subplots):
                            ax[j].set_xlabel('t in ns', fontsize=self.plot_label_size)
                            if grid:
                                ax[j].grid()
                            if legend:
                                ax[j].legend(bbox_to_anchor=(0.97, 1.0), loc=1)
                            if y_labels != None:
                                ax[j].set_ylabel(y_labels[j], fontsize=self.plot_label_size)
                    else:
                        for i, key in enumerate(plot_idxs):
                            ax.plot(data[k]['time'] * 1e9, scaling[i] * data[k][key])
                        # general formatting of all axes
                        if y_labels != None:
                            ax.set_ylabel(y_labels[0], fontsize=self.plot_label_size)
                        ax.set_xabel('t in ns')
            if no_frame:
                if type(ax) == np.ndarray:
                    for a in ax:
                        a.spines["top"].set_visible(False)
                        a.spines["right"].set_visible(False)
                else:
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
            plt.tight_layout()
            if save_fig:
                if fname != None:
                    self.plot_fname = fname
                plt.savefig(f'{self.plot_fname}.png')
                print(f'saved plot to {self.plot_fname}.png')
                plt.close()
            else:
                plt.show()
        elif mode == 'raster':
            if plot_idxs is not None:
                data = [self.verification[index] for index in plot_idxs]
            else:
                data = self.verification

            if time_frame != None:
                assert len(time_frame)==1, 'Not implemented: start and end point for time fram in raster plot'
                # dt = value_string_to_float(self.step_size)
                dt = 1e-15
                for i in range(len(data)):
                    d_array = np.array(data[i])
                    d_array_frame = d_array[d_array > time_frame[0]*dt]
                    data[i] = d_array_frame



            data = [np.array(data[k]) * 1e9 for k in range(len(data))]  # convert time-scale to ns
            ax = raster(data, color='k')
            ax.set_xlabel('t (ns)', fontsize=self.plot_label_size)
            if self.plot_latex:
                ax.set_ylabel('neuron index', fontsize=self.plot_label_size)
            else:
                ax.set_ylabel('# neurons', fontsize=self.plot_label_size)
            if title:
                ax.set_title('Spike raster plot', fontsize=self.plot_title_size)
            if save_fig:
                if fname != None:
                    self.plot_fname = fname
                plt.savefig(f'{self.plot_fname}_raster.png')
                print(f'saved raster plot to {self.plot_fname}_raster.png')
                plt.close()
            else:
                plt.show()

        else:
            raise NotImplementedError(f'Plot mode = {mode} is not implmeneted please chose another one!')

    def get_time_series(self, idxs=None, key=None):
        """
        Function the returns specific time series from the output file
        :param idxs: int or list
                     indices of the ime series, can be a list or a single number
        :param key: string,
                    key name of the time series
        :return: out, np.ndarray
                array of output time series read from the panda dataframe
        """
        data = pd.read_csv(self.out_fname, sep=',')
        keys = data.keys()
        if idxs != None:
            out = data[keys[idxs]]
        elif key != None:
            out = data[key]
        else:
            raise NotImplementedError('Please specify idxs or key')
        return out

    def fft(self, idxs=None, mean=True, stft=False, save_fig=True):

        """
        Function that computes the FFT or STFT for the voltage time courses of JJs
        :param idxs:
        :param mean:
        :return:
        """
        # read the output file and record step times
        df = pd.read_csv(self.out_fname, sep=',')

        # extract time as first row in .csv
        keynames = df.keys()
        t = df['time']
        val_keys = keynames[1:]

        if idxs != None:
            val_keys = val_keys[idxs]
        # extract remaining values and format to array
        key_length = len(val_keys)
        values = []
        value_names = []
        for i in range(1, key_length):
            if 'v(' in val_keys[i].lower():
                values.append(df[val_keys[i]])
                value_names.append(val_keys[i])

        values = np.array(values)

        if not mean:
            for i in range(values.shape[0]):
                if not stft:
                    sample_rate = 1 * np.mean(np.ediff1d(t))
                    N = t.shape[0]
                    yf = np.abs(fft.rfft(values))
                    xf = fft.rfftfreq(N, 1 / sample_rate)

                    plt.scatter(xf, yf, s=0.1)
                    plt.title('FFT of voltage')
                    plt.ylabel('Amplitude')
                    plt.xlabel('f in Hz')
                    plt.yscale('log')
                    plot_name = self.plot_fname + '_' + value_names[i] + '_fft.png'

                else:
                    # sampling frequency
                    sample_rate = 1 * np.mean(np.ediff1d(t))
                    fs = 1 / sample_rate
                    f, t, Zxx = signal.stft(values, fs, nperseg=256)
                    plt.pcolormesh(t * 1e9, f, np.abs(Zxx), vmin=0, vmax=np.max(values), shading='gouraud')
                    plt.title('STFT Magnitude of v')
                    plt.ylabel('Frequency in Hz')
                    plt.xlabel('Time in ns')
                    plot_name = self.plot_fname + '_' + value_names[i] + '_stft.png'
                if save_fig:
                    plt.savefig(plot_name)
                    print(f'saved raster plot to {plot_name}')
                    plt.close()
                else:
                    plt.show()
        else:
            values_mean = np.mean(values, axis=0)
            if not stft:
                sample_rate = 1 * np.mean(np.ediff1d(t))
                N = t.shape[0]
                yf = np.abs(fft.rfft(values_mean))
                xf = fft.rfftfreq(N, 1 / sample_rate)

                plt.scatter(xf, yf, s=0.1)
                plt.title('FFT of voltage')
                plt.ylabel('Amplitude')
                plt.xlabel('f in Hz')
                plt.yscale('log')
                plot_name = self.plot_fname + '_mean_fft.png'

            else:
                # sampling frequency
                sample_rate = np.mean(np.ediff1d(t))
                fs = 1 / sample_rate
                f, t, Zxx = signal.stft(values_mean, fs, nperseg=256)
                plt.pcolormesh(t * 1e9, f, np.abs(Zxx), vmin=0, vmax=np.max(values_mean), shading='gouraud')
                plt.title('STFT Magnitude of v')
                plt.ylabel('Frequency in Hz')
                plt.xlabel('Time in ns')
                plot_name = self.plot_fname + '_mean_stft.png'
            if save_fig:
                plt.savefig(plot_name)
                print(f'saved raster plot to {plot_name}')
                plt.close()
            else:
                plt.show()

    def add_input(self, **input_kwargs):
        self.input.append(self.Input(Circuit=self, **input_kwargs))

    class Input():
        """
        Class that defines an input for the circuit (voltage, current or phase source)
        and writes it into the .cir file
        :param name: string
             name of the source, default: 'IIN'
        :param function: string
            type of the function (can only be pwl as of now) default: 'pwl'
        :param function_duration: float
            duration of the function, this will typically be altered during function creation, default: None
        :param kwargs:
            keyword arguments for the function that is applied from Utils.py
        """

        def __init__(self, Circuit, name='IIN', function='pwl', **kwargs):
            self.circuit = Circuit
            self.name = name
            self.function = function
            self.function_string = ''

            if function == 'pwl':
                pwl_kwargs = {
                    'pulse_width': 30,
                    'pulse_height': 600,
                    'pulse_separation': 150,
                    'number_of_pulses': 5,
                    'pulse_sustain': 20}
                pwl_kwargs.update(kwargs)
                # get the values for times and amps of the pwl function
                times, amps = pwl_function(**kwargs)
                # convert the time and amp values to a line of type string that calls the pwl function in josim
                self.function_string = convert_to_string_line(times=times, amps=amps, time_unit='p', amp_unit='u')
                # calculate duration of function
                self.function_duration = (pwl_kwargs['number_of_pulses'] + 1) * pwl_kwargs['pulse_separation']
            elif function == 'poisson-spikes':
                # poisson spikes are handeled as a pwl in josim
                self.function = 'pwl'

                # i_max = 300, rate = 1, mu = 0.008,
                # coeff_of_var = 0.5, t_start = 0.0, t_end = None,
                # delay = True, save = True
                poisson_kwargs = {
                    'ts': None,
                    'dt': 0.1,
                    'i_max': 1,
                    'rate': 1,
                    'radom_amp': False,
                    'mu': 1,
                    'coeff_of_var': 1,
                    't_start': 0.0,
                    't_end': None,
                    'delay': False,
                    'save': False}
                poisson_kwargs.update(kwargs)

                if type(poisson_kwargs['ts']) != np.ndarray:
                    raise ValueError('ts in "poisson-spikes()" kwargs must be np.ndarray!')

                # use higher virtual step size, to map spikes to it and sum spikes in shorter time steps over it
                # a 0 before and after each spike is needed to set up the pwl in josim later, for this the time-step
                # grid in the poisson process needs to be less precise
                dt_original = kwargs['dt']
                ts_original = kwargs['ts'].copy()
                dt_scale_factor = int(3)
                kwargs['dt'] = dt_scale_factor * kwargs['dt']
                kwargs['ts'] = np.arange(0.0, ts_original[-1] + dt_original, kwargs['dt'])
                # calculate the spikes
                low_res_amps = gen_poisson_spikes_input(**kwargs)[0]
                # map back to original dt by inserting zeros again, amps[::k] is used here (fill every k-th entry)
                amps = np.zeros_like(poisson_kwargs['ts'])
                amps[::dt_scale_factor] = low_res_amps
                # find non-zero entries
                non_zero_idxs = np.where(amps != 0)[0]
                # add value before and after non_zero_idxs to reset amplitudes to zero in pwl
                n_idxs = non_zero_idxs.shape[0]
                idxs = np.zeros(n_idxs * 3, dtype=int)
                idxs[:n_idxs] = non_zero_idxs
                for i, idx in enumerate(non_zero_idxs):
                    idxs[n_idxs + (2 * i)] = (idx - 1)
                    idxs[n_idxs + (2 * i + 1)] = (idx + 1)
                idxs.sort()
                pwl_amps = amps[idxs]
                # help with float formatting in string
                times = ts_original[idxs].round(8)
                self.function_string = convert_to_string_line(times=times, amps=pwl_amps, time_unit='p',
                                                              amp_unit='u')
                # add 0 0 at the start to properly init pwl in josim
                self.function_string = "(0 0 " + self.function_string[1:]
                self.function_duration = poisson_kwargs['ts'][-1]

            else:
                raise NotImplementedError("Please use an implemented function: 'pwl', 'poisson-spikes'!")

        def write_to_file(self):

            # reading the .cir file here and get lines to be altered
            with open(self.circuit.cir_fname, "r") as file:
                lines = file.readlines()
                # get contents of line of interest
                for line_idx, line in enumerate(lines):
                    if line[:len(self.name)] == self.name:
                        line_str = lines[line_idx]
                        keep_old_string_idx = line_str.find(self.function) + len(self.function)
                        # alter content with replacement_string
                        line_str_new = line_str[:keep_old_string_idx] + self.function_string
                        lines[line_idx] = line_str_new
                        print(f' changed line {line_idx + 1} in {self.circuit.cir_fname} to a new version of '
                              f'{self.name} {self.function}')
                # write to the .cir file
                with open(self.circuit.cir_fname, "w") as file:
                    file.writelines(lines)

        def update_Circuit_duration(self):
            """
            update the duration of the simulation such that it ends after the input function
            """
            simulation_duration = self.function_duration + self.circuit.fade_out_time
            transient_analysis_string = '.tran ' + self.circuit.step_size + ' ' + str(simulation_duration) + 'p ' + \
                                        str(self.circuit.output_start) + ' ' + str(self.circuit.output_step_size) + '\n'

            with open(self.circuit.cir_fname, "r") as file:
                lines = file.readlines()
                # get contents of line of interest
                for line_idx, line in enumerate(lines):
                    if line[:len('.tran ')] == '.tran ':
                        line_str = lines[line_idx]
                        # change line to new string
                        lines[line_idx] = transient_analysis_string
                with open(self.circuit.cir_fname, "w") as file:
                    file.writelines(lines)

    def get_simulation_time(self):
        """
        get start and end time from simulation file, .tran command line
        :return:
        """
        with open(self.cir_fname, "r") as file:
            lines = file.readlines()
            # get contents of line of interest
            for line_idx, line in enumerate(lines):
                if line.lower()[:len('.tran ')] == '.tran ':
                    transient_analysis_string = lines[line_idx]
        time_values = transient_analysis_string.split()
        if len(time_values) < 4:
            self.t_start = 0.0
        else:
            self.t_start = value_string_to_float(time_values[3])
        self.t_end = value_string_to_float(time_values[2])
        step_size_read = value_string_to_float(time_values[4])
        if self.step_size != step_size_read:
            # print(f'changed step_size to {step_size_read}!')
            self.step_size = step_size_read

        if type(self.time) != np.ndarray:
            self.time = np.arange(self.t_start, self.t_end, self.step_size)

    def restore_original_file(self):
        """
        Function that changes the read .cir file back to the initial state
        """
        if self.from_file:
            with open(self.cir_fname, "r") as file:
                lines = self.original_lines
            with open(self.cir_fname, "w") as file:
                file.writelines(lines)
        else:
            raise TypeError('No circuit file was used, it cannot be restored!')

    def build_circuit(self, subcircuits, subcir_fnames, port_list, input_name, output_value=1, out_ports=None,
                      circuit_name="Josim Circuit", t_end=None, step_size=0.2, outputs=[], parameters=None,
                      repeat_circuits=True, output_mode='voltage', noise=0):
        """
        Function that creates a circuit for a .cir file from incomming arguments like subcircuits, a port_lists,
        inputs and outputs
        :param subcircuits: list of strings
            List of names of the subcircuits
        :param subcir_fnames: list of strings
            List of filenames of the subcircuits
        :param port_list: list of list of int
            list of lists of ports that each subcircuit is connected to
        :param input_name: list of string or string
            Names of the input parts (sources) in the circuit I..., V... and P... first letters are interpreted as
            current, voltage or phase sources
        :param output_value:

        :param circuit_name: string
            name of the circuit that is stored in the header of the .cir file
        :param t_end: int
            end time for the simulation defined in the file
        :param step_size: int
            step size in ps for the simulation
        :param outputs: list of string
            parameters that are set to be print out (stored in a .csv file) in the .cir file
        :return:
        """

        # create headline for assembled circuit
        head_line = f"* {circuit_name} automatically assembled using build_circuit on {datetime.datetime.now()} \n \n"

        subcircuit_lines = []
        original_subcircuits = subcircuits.copy()
        for i, subcircuit in enumerate(subcircuits):
            # check if subcircuit was already made
            if repeat_circuits or not subcircuit in subcircuits[:i]:

                if repeat_circuits:
                    sub_cir_idx = str(i)
                else:
                    sub_cir_idx = ''

                if len(subcir_fnames) < i - 1:
                    cir_fname = subcir_fnames[i]
                else:
                    subcir_fnames.append(subcir_fnames[i - 1])
                    cir_fname = subcir_fnames[i]
                # reading the .cir file
                with open(cir_fname, "r") as file:
                    lines = file.readlines()

                # filter for the subcircuit in this file
                subcircuit_idx_start = 0
                subcircuit_idx_end = len(lines)
                for j in range(len(lines)):
                    if lines[j][:7] == ".SUBCKT" or lines[j][:7] == ".subckt":
                        if lines[j].find(subcircuit) > 1:
                            subcircuit_idx_start = j
                            index_loc = lines[j].find(subcircuit) + len(subcircuit)
                            lines[j] = lines[j][:index_loc] + sub_cir_idx + lines[j][index_loc:]
                    if lines[j][:5] == ".ENDS" or lines[j][:5] == ".ends":
                        if lines[j].find(subcircuit) > 1:
                            subcircuit_idx_end = j + 1
                            index_loc = lines[j].find(subcircuit) + len(subcircuit)
                            lines[j] = lines[j][:index_loc] + sub_cir_idx + lines[j][index_loc:]

                current_subcircuit = lines[subcircuit_idx_start:subcircuit_idx_end]
                subcircuit_lines.append(current_subcircuit)

                # check dependencies
                for i_line, line in enumerate(current_subcircuit):
                    # check for subcircuit calls by checking if line start is X...
                    if line[0].lower() == 'x':
                        subcircuit_name = line.split()[1]
                        if not subcircuit_name in subcircuits:
                            subcircuits.insert(i + 1, subcircuit_name)

                subcircuit_lines[-1].append('\n')

                if parameters != None:
                    if i in parameters['idx']:
                        p_idxs = [j for j, k in enumerate(parameters['idx']) if k == i]
                        for l in p_idxs:
                            subcircuit_lines[i] = self.set_value(parameter=parameters['parameters'][l],
                                                                 value=parameters['values'][l],
                                                                 type=parameters['types'][l],
                                                                 model_name=parameters['model_names'][l],
                                                                 subcircuit=subcircuit, cir_string=subcircuit_lines[i])

        # create main circuit
        main_circuit_lines = []
        main_circuit_lines.append('\n')
        main_circuit_lines.append('* main circuit\n')

        # INPUT
        # read input from self.input attributes
        input_function = [self.input[k].function + self.input[k].function_string for k in range(len(self.input))]
        if type(input_name) != list:
            input_name = [input_name]

        for i in range(len(input_function)):
            input_function_str = str(input_function[i])
            input_str = input_name[i] + ' ' + create_port_string([0, i + 1]) + input_function_str
            main_circuit_lines.append(input_str)

        main_circuit_lines.append('\n')

        # COMPONENTS
        for i, subcircuit in enumerate(original_subcircuits):
            if repeat_circuits:
                sub_cir_idx = str(i)
            else:
                sub_cir_idx = ''
            circuit_component = 'X' + str(i) + ' ' + subcircuit + sub_cir_idx + ' ' + create_port_string(
                port_list[i]) + '\n'
            main_circuit_lines.append(circuit_component)

        # OUTPUT AND NOISE
        if noise > 0:
            # insert noise in K
            noise_line = f'.TEMP {noise} \n'
            main_circuit_lines.append(noise_line)


        if output_mode == 'voltage' or output_mode == 'current':
            if out_ports != None:
                for i, port in enumerate(out_ports):
                    output_str = 'ROUT' + str(i) + ' ' + str(port) + ' 0 ' + str(output_value[i]) + '\n'
                    main_circuit_lines.append(output_str)
            else:
                output_str = 'ROUT ' + str(port_list[-1][-1]) + ' 0 ' + str(output_value) + '\n'
                main_circuit_lines.append(output_str)

        main_circuit_lines.append('\n')

        # evaluation
        output_lines = []
        output_lines.append('*circuit output \n')

        if t_end is None:
            # get maiximum function duration for input function
            n_inputs = len(self.input)
            input_durations = np.zeros(n_inputs)
            for i in range(n_inputs):
                input_durations[i] = self.input[i].function_duration
            max_function_duration = np.max(input_durations)
            t_end = max_function_duration + self.fade_out_time

        transient_analysis_string = '.tran ' + str(step_size) + 'p ' + str(t_end) + 'p 0 ' + str(step_size) + 'p\n'
        output_lines.append(transient_analysis_string)
        for i in range(len(outputs)):
            output_lines.append('.print ' + outputs[i] + '\n')

        output_lines.append('.ends')

        with open(self.cir_fname, 'w') as file:
            file.write(head_line)
            for i, _ in enumerate(subcircuit_lines):
                file.writelines(subcircuit_lines[i])
            file.writelines(main_circuit_lines)
            file.writelines(output_lines)

        self.temp_cir_fname = self.cir_fname

    def build_subcircuit(self, subcir_fname, source_file=None, cir_type='splitter', parameters=None, name=None,
                         ports=[]):
        """
        This function is merely a hotfix-amalgamation of what I need!
        :param subcir_fname: Name of the subcircuit file
        :param source_file: name of the source circuit
        :param cir_type: type of the circuit
        :param parameters: dictionary of parameters
        :param name: name of new subcicuit
        :param ports: in-out port connections
        """

        # create headline for assembled circuit
        head_line = f"* {subcir_fname} automatically assembled using build_subcircuit on {datetime.datetime.now()} \n \n"

        # create subcircuit
        subcircuit_lines = []

        # read-in source circuit
        if cir_type == 'splitter':

            cir_fname = '2-Split'
            # reading the .cir file
            with open(source_file, "r") as file:
                lines = file.readlines()

            # filter for the subcircuit in this file
            subcircuit_idx_start = 0
            subcircuit_idx_end = len(lines)
            for j in range(len(lines)):
                if lines[j][:7] == ".SUBCKT" or lines[j][:7] == ".subckt":
                    if lines[j].find(cir_fname) > 1:
                        subcircuit_idx_start = j
                        index_loc = lines[j].find(cir_fname) + len(cir_fname)
                        lines[j] = lines[j][:index_loc] + lines[j][index_loc:]
                if lines[j][:5] == ".ENDS" or lines[j][:5] == ".ends":
                    if lines[j].find(cir_fname) > 1:
                        subcircuit_idx_end = j + 1
                        index_loc = lines[j].find(cir_fname) + len(cir_fname)
                        lines[j] = lines[j][:index_loc] + lines[j][index_loc:]

            source_cir_lines = lines[subcircuit_idx_start:subcircuit_idx_end]

            source_cir_lines[0] = source_cir_lines[0][:-1]

            n = parameters['n']
            # 570uA was used to bias two junctions
            IB_value = 570 / 2 * n

            l_idx = 7
            jj_idx = 3
            port_idx = 8
            l_line_idx = 8
            jj_line_idx = 13

            for i in range(n - 2):
                l_idx += 1
                port_idx += 1
                L1_str = f'L{l_idx}\t4\t{port_idx}\t1.65pH\n'
                l_line_idx += 1
                source_cir_lines.insert(l_line_idx, L1_str)

                l_idx += 1
                port_idx += 1
                L2_str = f'L{l_idx}\t{port_idx - 1}\t{port_idx}\t1.9pH\n'
                l_line_idx += 1
                source_cir_lines.insert(l_line_idx, L2_str)

                jj_idx += 1
                JJ_str = f'B{jj_idx}\t{port_idx - 1}\t0\tjsplit\n'
                jj_line_idx += 3
                source_cir_lines.insert(jj_line_idx, JJ_str)

                source_cir_lines[0] = str(source_cir_lines[0]) + f'\t{port_idx}'

            source_cir_lines[-4] = f'IB\t0\t3\t{IB_value}uA'
            source_cir_lines[0] = str(source_cir_lines[0]) + '\n'

        elif cir_type == 'sub-sub':

            # get subcircuits
            sub_subcircuit_lines = []
            for i, subcircuit in enumerate(parameters["subcircuits"]):
                if not subcircuit in parameters["subcircuits"][:i]:
                    cir_fname = parameters["subcircuit_fnames"][i]
                    # reading the .cir file
                    with open(cir_fname, "r") as file:
                        lines = file.readlines()

                    # filter for the subcircuit in this file
                    subcircuit_idx_start = 0
                    subcircuit_idx_end = len(lines)
                    for j in range(len(lines)):
                        if lines[j][:7] == ".SUBCKT" or lines[j][:7] == ".subckt":
                            if lines[j].find(subcircuit) > 1:
                                subcircuit_idx_start = j
                                index_loc = lines[j].find(subcircuit) + len(subcircuit)
                                lines[j] = lines[j][:index_loc] + lines[j][index_loc:]
                        if lines[j][:5] == ".ENDS" or lines[j][:5] == ".ends":
                            if lines[j].find(subcircuit) > 1:
                                subcircuit_idx_end = j + 1
                                index_loc = lines[j].find(subcircuit) + len(subcircuit)
                                lines[j] = lines[j][:index_loc] + lines[j][index_loc:]

                    sub_subcircuit_lines.append(lines[subcircuit_idx_start:subcircuit_idx_end])
                    sub_subcircuit_lines[-1].append('\n')

                    if parameters != None:
                        if i in parameters['idx']:
                            p_idxs = [j for j, k in enumerate(parameters['idx']) if k == i]
                            for l in p_idxs:
                                sub_subcircuit_lines[-1] = self.set_value(parameter=parameters['parameters'][l],
                                                                          value=parameters['values'][l],
                                                                          type=parameters['types'][l],
                                                                          model_name=parameters['model_names'][l],
                                                                          subcircuit=subcircuit,
                                                                          cir_string=sub_subcircuit_lines[-1])

            # build new subcircuit from existing subcircuits
            source_cir_lines = []

            # set up ports for new complete circuit
            port_str = ''.join(str(x) + '\t' for x in ports)
            # set up start and end lines
            start_line = f'.SUBCKT {name}\t{port_str}\n \n'
            source_cir_lines.append(start_line)
            end_line = f'.ends {name}\n'
            # add circuit components
            for i, subcircuit in enumerate(parameters["subcircuits"]):
                connect_str = ''.join(str(x) + '\t' for x in parameters["sub_ports"][i])
                line_i = f'X{i}\t' + subcircuit + '\t' + connect_str + '\n'
                source_cir_lines.append(line_i)
            source_cir_lines.append(end_line)

        subcircuit_lines += source_cir_lines

        # build new file that cointains the new subcircuit
        with open(subcir_fname, 'w') as file:
            file.write(head_line)
            if parameters != None:
                if "subcircuits" in parameters.keys():
                    for i, _ in enumerate(sub_subcircuit_lines):
                        file.writelines(sub_subcircuit_lines[i])
            file.writelines(subcircuit_lines)

        self.temp_files.append(subcir_fname)

    def clear_output(self, exclude=[]):
        """ Function to clear temporary output files
        :param exclude: list of string
            list of Circuit attributes that are excluded from being deleted """
        if not 'out_fname' in exclude:
            os.remove(self.out_fname)
        if not 'temp_cir_fname' in exclude and self.temp_cir_fname is not None:
            os.remove(self.temp_cir_fname)
        if not 'temp_files' in exclude:
            if len(self.temp_files) > 0:
                for file in self.temp_files:
                    try:
                        os.remove(file)
                    except:
                        print(f'WARNING: temp_file {file} not found!')

    def run_simulation_set(self, param_dict, out_idxs=None):

        if 'output_variable' in param_dict:
            vname = param_dict['output_variable']
        else:
            vname = None

        # test if parameter values are negative
        # raise error if it fails
        if np.min(param_dict["parameter_values"]) < 0:
            raise ValueError(f'param_dict["parameter_values"] contains negative parameter values \n process aborted!')

        N = param_dict['parameter_values'][0].shape[0]
        output_fname = self.out_fname
        out_values = []
        out_shapes = []

        print_date_time()
        for j in tqdm(range(N), desc="progress"):
            if j == 0:
                print("\n")  # handel tqmd bug with printout formatting
            cir_fn_out = self.cir_fname[:-4] + str(j) + '.cir'
            for i in range(len(param_dict['parameters'])):
                value_ij = param_dict['parameter_values'][i][j]
                str_value_ij = str(value_ij) + param_dict['units_prefix'][i]
                if i == 0:
                    cir_fname_current = self.cir_fname
                else:
                    cir_fname_current = cir_fn_out
                if 'subcircuit' in param_dict.keys():
                    subcircuit = param_dict['subcircuit'][i]
                else:
                    subcircuit = None

                self.set_value(cir_fname=cir_fname_current, parameter=param_dict['parameters'][i], value=str_value_ij,
                               type=param_dict['types'][i], model_name=param_dict['model_names'][i],
                               subcircuit=subcircuit, fn_out=cir_fn_out)

            if vname != None:
                self.run_simulation(temp_cir_fname=cir_fname_current, temp_out_fname=output_fname, spec_needed=False,
                                    verbose=False)
                out_vals_j = np.array(extract_value(fname=output_fname, vname=vname)[1])  # skip time values
                # out_vals_j[np.where(out_vals_j == 0.0)] = 1e-30 # fix problem with first time step being 0
                out_values.append(out_vals_j)
                out_shapes.append(out_values[j].shape[0])
                # not needed anymore ?
                # if out_values[j][0] == 0:  # fix problem with first time step being 0
                #         out_values[j] = out_values[j][1:]
            elif out_idxs != None:
                step_times = self.run_simulation(temp_cir_fname=cir_fname_current, temp_out_fname=output_fname,
                                                 spec_needed=True, index_list=out_idxs)
                out_values.append(step_times)

            else:
                raise ValueError('Please specifiy output_variable for time series or output_index '
                                 'in an input dictionary for margin analysis!')

            # remove current cir file after usage
            os.remove(cir_fname_current)

        # remove temporary file
        os.remove(output_fname)

        if len(out_shapes) > 0:
            max_shape = np.max(np.array(out_shapes))
            for i in range(len(out_values)):
                if out_values[i].shape[0] == max_shape - 1:
                    out_values[i] = np.hstack([np.array([np.random.rand() * 1e-10]), out_values[i]])

            return np.array(out_values)
        else:
            return out_values

    def set_value(self, parameter, value, cir_fname=None, type='default', model_name=' ', subcircuit=None, fn_out=None,
                  line=None, cir_string=None):

        if cir_fname is None:
            cir_fname = self.cir_fname

        if cir_string is None:
            # reading the .cir file
            with open(cir_fname, "r") as file:
                lines = file.readlines()
        else:
            lines = cir_string

        if line != None:
            original_lines = copy.deepcopy(lines)
            lines = [lines[line]]

        param_length = len(parameter)
        if model_name is None:
            model_name = ''
        model_length = len(model_name)

        # map lines according to it being a subcircuit or not
        if subcircuit != None:
            subcircuit_idx_start = 0
            subcircuit_idx_end = len(lines)
            for i in range(len(lines)):
                if lines[i][:7] == ".SUBCKT" or lines[i][:7] == ".subckt":
                    if lines[i].find(subcircuit) > 1:
                        subcircuit_idx_start = i
                if lines[i][:5] == ".ENDS" or lines[i][:5] == ".ends":
                    if lines[i].find(subcircuit) > 1:
                        subcircuit_idx_end = i
            original_lines = copy.deepcopy(lines)
            lines = lines[subcircuit_idx_start:subcircuit_idx_end]

        # find line that corresponds to parameter

        value_found = False

        if type == 'model_param':
            line_start = '.model ' + model_name
            l_s_length = len(line_start)
            for i in range(len(lines)):
                if lines[i].strip()[:l_s_length].lower() == line_start.lower():
                    # in model <parameter>=<value><unit>,  must occur that ends either with , or with )
                    # add +1 for the '=' expression after the parameter name
                    value_start = lines[i].find(parameter) + param_length + 1
                    line_words = lines[i].split()
                    has_parameter = [string_i.count(parameter) for string_i in line_words]
                    parameter_in_lines_index = has_parameter.index(1)
                    parameter_string = line_words[parameter_in_lines_index]
                    value_end = parameter_string.rfind(value[-1])
                    num_value = get_num_from_string(value)
                    parameter_string_new = parameter_string[:param_length] + '=' + str(num_value) + parameter_string[
                                                                                                    value_end:]
                    line_words[parameter_in_lines_index] = parameter_string_new
                    # replace parameter value
                    lines[i] = ' '.join(line_words) + '\n'
                    value_found = True
        elif type == 'default':
            for i in range(len(lines)):
                if lines[i].strip()[:param_length].lower() == parameter.lower():
                    value_start, value_len = find_last_word_idxs(lines[i])
                    # delete parameter value
                    lines[i] = lines[i][:value_start]
                    # write new parameter value
                    lines[i] = lines[i] + value + '\n'
                    value_found = True
        elif type == 'model':
            # here the model_name is the name of the element in the circuit and the parameter name ist the actual parameter
            for i in range(len(lines)):
                if lines[i].strip()[:model_length].lower() == model_name.lower():
                    if lines[i].lower().find(parameter.lower()) > -1:
                        # add +1 for the '=' expression after the parameter name
                        value_start = lines[i].find(parameter) + param_length + 1
                        # delete parameter value
                        last_comma_place = lines[i].rfind(',')
                        end_string = lines[i][value_start:]
                        if last_comma_place < value_start:
                            # find the space after the value begins which should be the value end
                            value_end = value_start + find_end_of_value(end_string)
                        else:
                            value_end = last_comma_place
                        # replace parameter value
                        lines[i] = lines[i][:value_start] + value + lines[i][value_end + 1:]
                        value_found = True
                    else:
                        if parameter.lower() == 'icrit':
                            parameter = 'ic'
                        lines[i] = lines[i][:-1] + ' ' + parameter + '=' + value + '\n'
                        value_found = True
        else:
            raise ValueError('Please select a valid model from: {default, model, model_param}!')

        if not value_found:
            warnings.warn(f'Warning! for type {type} with parameter {parameter}'
                          f' was no value found in file {cir_fname}')
            print(f'>>Warning! for type {type} with parameter {parameter}'
                  f' no value was found in file {cir_fname}')

        # formatting for output
        lines_out = lines

        if line != None:
            original_lines[line] = lines[0]
            lines_out = original_lines

        if subcircuit != None:
            original_lines[subcircuit_idx_start:subcircuit_idx_end] = lines
            lines_out = original_lines

        if cir_string is None:
            # write to the .cir file
            if fn_out is None:
                fn_out = cir_fname
            with open(fn_out, "w") as file:
                file.writelines(lines_out)
        else:
            lines.append(cir_string[-2])
            lines.append(cir_string[-1])
            return lines

    def get_value(self, parameter, type='default', model_name=' ', subcircuit=None, line=None):

        value = None
        # reading the .cir file
        with open(self.cir_fname, "r") as file:
            lines = file.readlines()

        if line != None:
            lines = [lines[line]]

        # map lines according to it being a subcircuit or not
        if subcircuit != None:
            subcircuit_idx_start = 0
            subcircuit_idx_end = len(lines)
            for i in range(len(lines)):
                if lines[i][:7].lower() == ".subckt":
                    if lines[i].lower().find(subcircuit.lower()) > 1:
                        subcircuit_idx_start = i
                if lines[i][:5].lower() == ".ends":
                    if lines[i].lower().find(subcircuit.lower()) > 1:
                        subcircuit_idx_end = i

            lines = lines[subcircuit_idx_start:subcircuit_idx_end]

        param_length = len(parameter)
        if model_name is None:
            model_name = ' '
        model_length = len(model_name)
        # find line that corresponds to parameter
        if type == 'model_param':
            line_start = '.model ' + model_name
            l_s_length = len(line_start)
            for i in range(len(lines)):
                if lines[i].lower().strip()[:l_s_length] == line_start.lower():
                    # find parameter in string
                    value = get_parameter_value(lines[i], parameter)
        elif type == 'default':
            for i in range(len(lines)):
                if lines[i].lower().strip()[:param_length] == parameter.lower():
                    value_start, value_len = find_last_word_idxs(lines[i])
                    value = lines[i][value_start:(value_start + value_len)]
        elif type == 'model':
            # here the model_name is the name of the element in the circuit and the parameter name ist the actual parameter
            for i in range(len(lines)):
                if lines[i].lower().strip()[:model_length] == model_name.lower():
                    # add +1 for the '=' expression after the parameter name
                    value_start = lines[i].find(parameter) + param_length + 1
                    # delete parameter value
                    last_comma_place = lines[i].rfind(',')
                    end_string = lines[i][value_start:]
                    if last_comma_place < value_start:
                        # find the space after the value begins which should be the value end
                        value_end = value_start + find_end_of_value(end_string)
                    else:
                        value_end = last_comma_place
                    # replace parameter value
                    value = lines[i][value_start:value_end + 1]
                    if value == '':
                        # look for model if no original value is given in the line where the model is called
                        words = lines[i].split()
                        model_name = words[-1]
            if value == '':
                # find actual model that is referred to in the line
                line_start = '.model ' + model_name
                l_s_length = len(line_start)
                for i in range(len(lines)):
                    if lines[i].lower().strip()[:l_s_length] == line_start.lower():
                        # find parameter in string
                        value = get_parameter_value(lines[i], parameter)

        else:
            raise ValueError('Please select a valid model from: {default, model, model_param}!')

        return value

    def get_kuramoto(self, parameter_idxs=None, time_frame=None, plot=False, save_fig=False, undersample=1000,
                     avg=False, grid=True, title=True, no_frame=True):

        # read the output file and record step times
        df = pd.read_csv(self.out_fname, sep=',')

        # extract time as first row in .csv
        keynames = df.keys()
        time_vals = df['time']
        val_keys = keynames[1:]

        if parameter_idxs != None:
            val_keys = val_keys[parameter_idxs]
        # extract remaining values and format to array
        key_length = len(val_keys)
        values = []
        n_phases = 0
        for i in range(1, key_length):
            if 'p(' in val_keys[i].lower():
                values.append(df[val_keys[i]])
                n_phases += 1

        values = np.array(values)

        if undersample == None:
            sample_rate = 1
        else:
            sample_rate = int(np.floor(time_vals.shape[0] / undersample))
        if sample_rate == 0:
            sample_rate = 1
        values = values[:, ::sample_rate]
        time_vals = time_vals[::sample_rate]

        # convert phases to 2pi
        phases = values % (2 * np.pi)

        osc_signal = np.exp(1j * phases)

        if time_frame != None:
            if len(time_frame) == 1:
                osc_signal = osc_signal[:, time_frame[0]:]
                time_vals = time_vals[time_frame[0]:]
            else:
                osc_signal = osc_signal[:, time_frame[0], time_frame[1]]
                time_vals = time_vals[time_frame[0], time_frame[1]]

        order_param = 1 / n_phases * np.sum(osc_signal, axis=0)
        coherence_parameter = np.abs(order_param)
        avg_phase = np.angle(order_param)
        mean_coh = np.mean(coherence_parameter)

        if plot:

            # plot_time = np.linspace(t_start, t_end, 1000)
            # coherence_parameter_plot = np.interp(x=plot_time, xp=time_vals, fp=coherence_parameter)
            fig, ax = plt.subplots(1, 1)
            ax.plot(time_vals * 1e9, coherence_parameter)
            ax.set_xlabel('time (ns)', fontsize=self.plot_label_size)
            ax.set_ylabel('Kuramoto coherence parameter', fontsize=self.plot_label_size)
            ax.set_xlim(np.array(time_vals)[0] * 1e9, np.array(time_vals)[-1] * 1e9)
            if no_frame:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            if title:
                ax.set_title(f'a = {mean_coh:.5f}', fontsize=self.plot_title_size)
            if grid:
                plt.grid()
            if save_fig:
                plt.savefig(self.plot_fname + "_kuramoto.png")
                print(f'saved kuramoto plot to {self.plot_fname}_kuramoto.png')
                plt.close()
            else:
                plt.show()

        # apply mean to output values if specified
        if avg:

            import math
            if math.isnan(mean_coh):
                print('r is NaN! Formatting is still wrong!')

            coherence_parameter = mean_coh
            avg_phase = np.mean(avg_phase)

        return coherence_parameter, avg_phase

    def load_verification(self, index_list=None, verbose=False, save=True, fname=None):

        if fname == None:
            if self.data_fname[-4:] == '.csv':
                self.data_fname = self.data_fname[:-4]
            fname = f'{self.data_fname}_verification.csv'

        if not os.path.exists(fname):
            # read the output file and record step times
            df = pd.read_csv(self.out_fname, sep=',')

            # extract time as first row in .csv
            keynames = df.keys()
            time_vals = df['time']
            val_keys = keynames[1:]
            self.output_keys = val_keys

            # extract remaining values and format to array
            key_length = len(val_keys)
            values = []
            for i in range(0, key_length):
                if 'p(' in val_keys[i].lower():
                    values.append(df[val_keys[i]])

            step_times = []
            if index_list == None:
                index_list = np.arange(len(values))
            if verbose:
                for i in tqdm(index_list, f'loading verification from {len(values)} values'):
                    # steps = get_steps(values[i])
                    step_mask = get_Steps_mask(values[i], noise_tolerance=0)
                    steps = np.where(step_mask == 1)[0]
                    step_times.append(time_vals[steps].to_numpy())
            else:
                for i in index_list:
                    # steps = get_steps(values[i])
                    step_mask = get_Steps_mask(values[i], noise_tolerance=0)
                    steps = np.where(step_mask == 1)[0]
                    step_times.append(time_vals[steps].to_numpy())
            self.verification = step_times
            self.time = time_vals

            if save:
                print(f'saved verification data to {fname}')
                with open(fname, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(self.verification)

                df.to_csv(self.data_fname + '.csv')
                # with h5py.File(f'{self.data_fname}_verification.hdf5', "w") as h5file:
                #     h5file.create_dataset('verification', data=self.verification)
                # self.temp_files.append(f'{self.data_fname}_verification.hdf5')
        else:
            print(f'loading verification from {fname}')

            with open(fname, 'r') as f:
                reader = csv.reader(f)
                self.verification = list(reader)

            for k, _ in enumerate(self.verification):
                for l, _ in enumerate(self.verification[k]):
                    self.verification[k][l] = float(self.verification[k][l])

            if os.path.exists(self.out_fname) and type(self.time) != np.ndarray:
                df = pd.read_csv(self.out_fname, sep=',')
                keynames = df.keys()
                time_vals = df['time']
                self.time = time_vals
            # with h5py.File(fname, 'r') as h5file:
            #     self.verification = np.array(h5file['verification'])






