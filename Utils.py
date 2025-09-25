import numpy as np
import pandas as pd
import re
import scipy
import sys
import os
import uuid
import random
import h5py
from tqdm import tqdm
from scipy.special import gamma
from contextlib import contextmanager
import datetime
import matplotlib
import matplotlib.pyplot as plt


def get_steps(data, threshold=np.pi):
    """
    find steps in a time series, that consecutively cross the multiple of a threshold
    if x = f(t) > x_thres at t_i and x > 2*x_thres at t_j then steps = [i, j] are recorded
    :param data:, np.ndarray, input time series
    :param threshold:, float, threshold value, default=pi
    :return steps: np.ndarray, indices of steps in timeseries
    """
    steps = []
    start_vals = data[:100]
    base_line = np.mean(start_vals)
    i = 0
    while i < data.shape[0]:
        if data[i] > (base_line + threshold):
            steps.append(i)
            base_line += 2 * threshold
        i += 1
    return np.array(steps)

import numpy as np

def get_Steps_mask(time_series, threshold=2*np.pi, noise_tolerance=0.3*np.pi):
    # Ensure the input is a numpy array
    time_series = np.array(time_series)

    # Initialize a mask with zeros
    mask = np.zeros_like(time_series, dtype=int)

    # Initialize the reference level
    base_line = np.mean(time_series[0:100])

    # Loop through the time series and detect jumps
    for i in range(1, time_series.shape[0]):
        if np.abs(time_series[i] - base_line) > threshold - noise_tolerance:
            mask[i] = 1
            base_line = np.mean(time_series[i-100:i])

    return mask

def pwl_function(pulse_width, pulse_height, pulse_separation, number_of_pulses, pulse_sustain=0):
    """
    Creates two numpy arrays with time and amp values for a pwl
    :param pulse_width: float
        width of each pulse
    :param pulse_height: float
        height of each pulse
    :param pulse_separation: float
        separation between pulses
    :param number_of_pulses: float
        number of pulses that are done
    :param pulse_sustain: float
        time that the peak of a pulse is sustained. If this is 0, then the values ramp up to the peak
        and then ramp down again from the peak.
    :return: pulse_times, np.ndarray
        times for each pulse
    :return: pulse_amplitudes: np.ndarray
        amplitudes for each pulse
    """
    pulse_amplitudes = np.zeros((4*number_of_pulses)+1)
    pulse_times = np.zeros_like(pulse_amplitudes)
    for i in range(number_of_pulses):
        j = 1 + 4*i
        pulse_times[j] = pulse_separation*(i+1)
        pulse_times[j+1] = pulse_separation * (i+1) + ((pulse_width - pulse_sustain)/2)
        pulse_times[j+2] = pulse_separation * (i+1) + pulse_width - ((pulse_width - pulse_sustain) / 2)
        pulse_times[j+3] = pulse_separation * (i+1) + pulse_width

        pulse_amplitudes[j] = 0
        pulse_amplitudes[j + 1] = pulse_height
        pulse_amplitudes[j + 2] = pulse_height
        pulse_amplitudes[j + 3] = 0
    return pulse_times, pulse_amplitudes

def convert_to_string_line(times, amps, time_unit='p', amp_unit='u'):
    """
    Function that converts two arrays into a string as line in a .cir file
    The idea is to use this to convert the values from a pwl function to a line
    that is later stored in a .cir file.

    :param times: np.ndarray
        array of time values
    :param amps: np.ndarray
        array of amplitude values
    :param time_unit: string
        unit prefix for the time that is used in josim, default is 'p' for pico
    :param amp_unit: string
        unit prefix for the amplitude that is used in josim, default is 'u' for micro
    :return:
        string with (... values ...)

    Examples usage
    --------
    >>> times, amps = pwl_function(pulse_width=30, pulse_height=600, pulse_separation=150, number_of_pulses=4,
    >>>                            pulse_sustain=20)
    >>> pwl_string = 'pwl' + Convert_to_string_line(times=times, amps=amps, time_unit='p', amp_unit='u')

    """
    line = []
    for a, b in zip(times, amps):
        # Convert entries to strings and add a '.' at the end if they are numbers
        time = f"{a}"+time_unit if isinstance(a, (int, float)) else str(a)
        amp = f"{b}"+amp_unit if isinstance(b, (int, float)) else str(b)
        # Append the formatted entries to the result list

        # use 0 instead of 0.0 just in case int to float conversion is an issue in josim
        if a == 0.0:
            line.append('0')
        else:
            line.append(time)
        if b == 0.0:
            line.append('0')
        else:
            line.append(amp)

    return '(' + " ".join(line) + ')\n'

def create_port_string(ports):
    """Function that creates a string from a list of list of ports for subcircuits
    :param ports: list of lists of int
        list with lists of all ports to which a subcircuit is connected
    :return port_str: string
        string version of this list
    """
    port_str = ''
    for k in range(len(ports)):
        port_str += (str(ports[k]) + ' ')
    return port_str

def t_format(time):
    """
    Function that formats a float value to a nicely readable time format
    :param time: float
        float value that represents time in seconds
    :return: t, unit: float, string
        tuple of a formated time in seconds, minutes, hours or days and the unit it is in as a string
    """
    if time < 60:
        return time, "s"
    else:
        t_min = time / 60
        if t_min < 60:
            return t_min, "min"
        else:
            t_h = t_min / 60
            if t_h < 24:
                return t_h, "h"
            else:
                t_d = t_h / 24
                return t_d, "d"

def extract_value(fname, vname):
    """
    function that reads out a time series and the corrsponding time values from a pd.Dataframe
    :param fname: string
        filename of a .csv where the data is stored
    :param vname: string
         name of the variable the is extracted
    :return: time_vals, out_values: np.ndarry, np.ndarray
        tuple of two arrays with the time values and the variable values at those time points
    """
    df = pd.read_csv(fname, sep=',')
    keynames = df.keys()
    time_vals = df[keynames[0]]
    out_values = df[vname]
    return time_vals, out_values

def value_string_to_float(string):
    """Converts string with standard value units to a float with specified unit
    :param string
        number with unit suffix to be formated to float
    :return number: float
        formated number as correct float value"""
    # check if unit of physical quantity is used and ignore
    if len(string) > 1:
        if not string[-2].isdigit() and string[-2] != '.':
            string = string[:-1]

    # init exponent
    exp = 1
    suffix = string[-1]

    # check if a suffix is used or not
    if not suffix.isdigit():
        if suffix == 'm':
            exp = 1e-3
        elif suffix == 'u':
            exp = 1e-6
        elif suffix == 'n':
            exp = 1e-9
        elif suffix == 'p':
            exp = 1e-12
        elif suffix == 'f':
            exp = 1e-15
        elif suffix == 'a':
            exp = 1e-18
        elif suffix == 'k':
            exp = 1e3
        elif suffix == 'M':
            exp = 1e6
        elif suffix == 'G':
            exp = 1e9
        elif suffix == 'T':
            exp = 1e12
        return float(string[:-1])*exp
    else:
        return float(string)


def get_parameter_value(search_string, parameter):
    """
    function that extracts the numeric value of a parameter of the type x=<num> from a string
    :param search_string:  string, string that is searched
    :param parameter: string, name of the parameter
    :return: value: float, numerical value that is extracted
    returns 0 if no value was found!
    """
    value = None
    has_parameter = [string_i.count(parameter) for string_i in search_string.split()]
    parameter_in_lines_index = has_parameter.index(1)
    parameter_string = search_string.split()[parameter_in_lines_index]
    int_value, float_value = re.findall('\d+', parameter_string), re.findall('\d+\.\d+', parameter_string)
    for v_i in (int_value, float_value):
        if len(v_i) == 1:
            value = float(v_i[0])
    if value == None:
        raise ValueError(f'No value was found for {parameter}!')
    return value

def find_end_of_value(txt):
    """
    Function that looks for the end of a value be searching for a space in the txt string
    :param txt: string
        input text
    :return: end_fo_value, int
     index of the end of the value
    """
    end_fo_value = txt.find(' ')
    if end_fo_value == - 1:
        end_fo_value = txt.find('\n')
    return end_fo_value - 1

def find_last_word_idxs(txt):
    """
    Function that finds the index of the start of the last word in a string
    as well as the length of this word
    :param txt: string
        input text
    :return: last_word_start, last_word_len: int, int
        tuple of the indexs where the last word starts and its length
    """
    txt_lower = txt.lower()
    words = txt_lower.split()
    last_word = words[-1].strip()
    last_word_start = txt_lower.rfind(last_word)
    last_word_len = len(last_word)
    return last_word_start, last_word_len

def get_num_from_string(searchstring):
    """
    Function that finds the first occurence of a number in a string
    :param searchstring: string
        string in which a number is searched
    :return: value: float
        number that was found
    """
    value = None
    int_value, float_value = re.findall('\d+', searchstring), re.findall('\d+\.\d+', searchstring)
    for v_i in (int_value, float_value):
        if len(v_i) == 1:
            value = float(v_i[0])
    if value == None:
        raise ValueError(f'No numerical was found in {searchstring}!')
    return value


def execute_silently(command, log=False):
    """
    Execute a command silently (suppressing output).
    :param log: Boolean
        option to keep the output of the function in a run_log.txt
    """
    run_id = str(uuid.uuid4())
    log_file = f'run_log_' + run_id + '.txt'
    os.system(command + '>' + log_file)
    if not log:
        os.remove(log_file)

def get_evr(x, y, t, weights=None, gauss=False):
    """
    calculate relative verification error over time
    :param x: input value
    :param y: reference value
    :param t: time
    :param weights: weights of input array in evaluation
    :param gauss: bool: usage of noise
    :return: e, np.ndarray of shape [ x ]
        error values
    """

    if type(x) is np.ndarray:
        x_len = x.shape[0]
    else:
        x_len = len(x)
    e = np.zeros(t.shape[0])

    if weights is None:
        weights = np.ones(x_len)

    for i, t_i in enumerate(t):
        # calculate the sums of 2pi phase transitions for each out idx over time
        n_x = np.zeros(x_len)
        n_y = np.zeros_like(n_x)
        for j in range(x_len):
            n_x[j] = np.where(np.array(x[j]) < t_i)[0].shape[0] * weights[j]
            n_y[j] = np.where(np.array(y[j]) < t_i)[0].shape[0] * weights[j]
        N_x_y = np.sum(n_x==n_y)
        N_y = np.sum(n_y)
        e[i] = 1 - N_x_y / x_len
        if gauss:
            e[i] = scipy.ndimage.gaussian_filter1d(e[i], sigma=1)
    return e

def format_key_name(name_init):
    """
    Function that helps formating circuit keys
    :param name_init: str: initial name
    :return: newly formatted name
    """
    name_start = name_init.find('(') + 1
    name_end = name_init.find(')')
    name = name_init[name_start:name_end]
    name_parts = name.split('|')
    if len(name_parts) > 1:
        subcircuit_name = '|'.join(name_parts[1:])
        name = name_parts[0]
    else:
        subcircuit_name = ''
    return subcircuit_name + ' ' + name

def array_unflatten(array, n_rows=1):
    """ Unflattens an array of size N into nxm size with n_rows
    :param array: np.ndarray
        input array that needs unflattening
    :param n_rows, int
        number of rows that the new array should have
    :return new_array, np.ndarray
        unflattended array of size nxm
    """
    n_cols = int(array.shape[0] / n_rows)
    new_array = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        new_array[i, :] = array[(i * n_cols):((i + 1) * n_cols)]
    return new_array

def print_date_time():
    """
    Function that prints the date and time
    """
    date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(date)


def gen_poisson_spikes_input(ts, dt=0.1, i_max=300, rate=1,  random_amp=False, mu=0.008, coeff_of_var=0.5,
                             n_neurons=1, t_start=0.0, t_end=None, delay=True, save=True):
    """
    Generate spike times and currents for a neuron with a time-dependent firing rate using an inhomogeneous Poisson
     process.
    modified from:
    https://stackoverflow.com/questions/72970651/how-to-simulate-a-non-homogenous-poisson-process
    Parameters:
    imax (float): Max value of I which is used as scaling factor for random sampling
    rate (float): Firing rate at time t (spikes per second), can be function or float

    """

    if not callable(rate):
        rate_copy = rate
        rate = lambda x : rate_copy

    if t_end is None:
        t_end = ts[-1]

    i_s = np.zeros((n_neurons, ts.shape[0]))

    rate_max = np.max(rate(ts))
    # generate large reservoir of random values from gamma distribution
    if random_amp:
        scale = (coeff_of_var * mu) ** 2 / mu
        gamma_pdf = gamma(a=coeff_of_var ** (-2), loc=0, scale=scale)
        spike_values = gamma_pdf.rvs(int(1e5*rate_max))
    else:
        spike_values = np.ones(int(1e5*rate_max))

    rv_idx = 0
    for j in range(n_neurons):
        # The following should generate 5 cycles of non-zero
        # event epochs between time 0 and time 100
        t_vals = []
        t = 0.0
        while True:
            # generate Poisson candidate event times using
            # exponentially distributed inter-event delays
            # at the maximal rate

            if t > ts[-1]:
                break
            t += random.expovariate(rate_max)
            if random.random() <= rate(t) / rate_max:
                t_vals.append(t)

        t_grid, idxs = round_to_1dgrid(np.array(t_vals), ts, idx=True)
        t_length = len(t_vals)
        rv_idx += t_length
        # get all gamma values from range
        spike_vals = spike_values[rv_idx:t_length+rv_idx]
        # sum entries over grid points with values (gamma_values) as weights
        mapped_sums = np.bincount(idxs, weights=spike_vals)
        # get non-zero entries and unique idxs
        mapped_sums = mapped_sums[mapped_sums != 0]
        unique_idxs = np.unique(idxs)
        # update i values on these indexes with the non-zero entries scaled by imax
        i_s[j, unique_idxs] += mapped_sums * i_max

        # if delay:
        #         i_shape = i_s[j].shape[0] + self.alpha.shape[0] - 1
        #         i_delayed = np.zeros(i_shape)
        #         idxs = np.where(i_s[j] > 0)[0]
        #         for i_idx, idx in enumerate(idxs):
        #             i_delayed[idx:idx + self.alpha.shape[0]] += self.alpha * i_s[j, idx]
        #         i_s[j] = i_delayed[:ts.shape[0]]

    i_s[:, int(t_start / dt)] = 0
    i_s[:, int(t_end / dt):] = 0

    # if save:
    #     with h5py.File(self.fname + '.hdf5', 'w') as h5file:
    #         h5file.create_dataset('Iinj', data=self.Iinj)
    #     print(f'saved poisson input to {self.fname}.hdf5')

    return i_s

def round_to_1dgrid(x, grid, idx=False):
    """
    function that rounds input data x to closest points on a 1d grid
    :param x: input array
    :param grid: grid
    :return: mapped array
    """
    if not type(x) == np.ndarray:
        x = np.array([x])

    idxs = [find_closest(grid, val) for val in x]
    res = np.array(idxs)
    if idx:
        return res, np.array(idxs, dtype=int)
    else:
        return res

def find_closest(array, value):
    """
    function that finds closest idx of closest point in array to a given value
    :param array: array
    :param value: value
    :return: idx
    """
    return np.abs(array - value).argmin()


def add_component(comp_dict, components, dict_keys):
    """
    function that adds a component with various keys to a dictionary
    :param comp_dict: original dict where a component should be added to
    :param components: the components that should be added in a list or list of lists
    :param dict_keys: the keys of the dictionary
    """
    if type(components) is not list:
        components = [components]
    if isinstance(components[0], list):
        for i_comp, component in enumerate(components):
            if i_comp == 0:
                for i_key, key in enumerate(dict_keys):
                    comp_dict[key] = [component[i_key]]
            else:
                for i_key, key in enumerate(dict_keys):
                    comp_dict[key].append(component[i_key])
    else:
        for i_key, key in enumerate(dict_keys):
            # if key doesn't exist yet, make new else append
            if not key in comp_dict.keys():
                comp_dict[key] = [components[i_key]]
            else:
                comp_dict[key].append(components[i_key])

def raster(event_times_list, color='k'):
  """
  Creates a raster plot **with spikes saved at 300 dpi as raster art**

  Original code from https://gist.github.com/kylerbrown/5530238

  Parameters
  ----------
  event_times_list : iterable
                     a list of event time iterables
  color : string
          color of vlines

  Returns
  -------
  ax : an axis containing the raster plot

  This version attempts to rasterize the plot
  Reference: https://matplotlib.org/stable/gallery/misc/rasterization_demo.html
  """
  ax = plt.gca()
  for ith, trial in enumerate(event_times_list):
    plt.vlines(trial, ith + .5, ith + 1.5, color=color, rasterized=True)
  plt.ylim(.5, len(event_times_list) + .5)
  return ax


def t_format(time):
    """
    Function that creates a time value formatted to s, min, h or d
    :param time: float: time to be formatted
    :return: float, string: time, suffix of formated time
    """
    if time < 60:
        return time, "s"
    else:
        t_min = time / 60
        if t_min < 60:
            return t_min, "min"
        else:
            t_h = t_min / 60
            if t_h < 24:
                return t_h, "h"
            else:
                t_d = t_h / 24
                return t_d, "d"


def stretch_spikes(arr):
    """
    Function that holds spike inputs for a longer time duration
    :param arr: np.ndarray: array of spikes
    :return: results: np.ndarray: stretched array with steps instead of spikes
    """
    # Create an array to store the output
    result = np.copy(arr)
    # Find the indices where values are zero
    zero_indices = np.where(result == 0)[0]
    # Find the indices of non-zero values
    non_zero_indices = np.where(result != 0)[0]
    if len(non_zero_indices) == 0:
        return result
    # Fill zero values with the next non-zero value (moving backward)
    start = 0
    for i in non_zero_indices:
        result[start:i+1] = result[i]
        start = i
    return result