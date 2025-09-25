"""
This is an example of two different Neuron models (inhibitory and excitatory with a soma and a synapse component)
They model LIF point-like neurons and should allow to build an E/I balanced newtork with proper amplifications of the
output (splitter)
"""


import os
import numpy as np
from Circuit import Circuit
from Utils import add_component

out_fname = os.path.join('csv', 'temp.csv')
out_fname_1 = os.path.join('csv', 'temp_1.csv')

# create circuit from library of subcircuits
# define a new circuit from subcircuits and give it a new name
cir_name = f'temp.cir'
cir_fname = os.path.join('netlists', cir_name)
plot_fname = os.path.join('images', cir_name[:-4])
cir_name_1 = f'temp_1.cir'
cir_fname_1 = os.path.join('netlists', cir_name_1)
plot_fname_1 = os.path.join('images', cir_name_1[:-4])

# setting up circuit details
subcircuits = ["Neuron-Soma", "R-Synapse-exc"]
subcircuits_1 = ["Neuron-Soma", "R-Synapse-inh"]
subcircuit_fnames = ['netlists/Neuron_circuits.cir']*len(subcircuits)
port_list = [[0, 1, 2], [0, 2, 3]]
output_list = ['DEVI IIN1', 'PHASE B1.X0', 'DEVI ROUT', 'DEVV B1.X0']
input_names = ['IIN1']

# write parameter dict to change subcircuits
param_dict = dict()
param_dict_1 = dict()
param_keys = ['parameters', 'model_names', 'types', 'values', 'idx']
component_1exc = ['RSCALE', None, 'default', '20', 1]
component_1inh = ['RSCALE', None, 'default', '1e-4', 1]

add_component(param_dict, component_1exc, param_keys)
add_component(param_dict_1, component_1inh, param_keys)

# build the circuit using build_circuit
subs_circuit = Circuit(cir_fname=cir_fname, out_fname=out_fname, plot_fname=plot_fname, from_file=False,
                       fade_out_time=20)
subs_circuit_1 = Circuit(cir_fname=cir_fname_1, out_fname=out_fname_1, plot_fname=plot_fname_1, from_file=False,
                       fade_out_time=150)
step_size = 0.001

# STEP INPUT
# subs_circuit.add_input(function='pwl', pulse_width=50, pulse_height=600,
#                        pulse_separation=10, number_of_pulses=1, pulse_sustain=480)
# subs_circuit.add_input(function='pwl', pulse_width=500, pulse_height=400,
#                        pulse_separation=50, number_of_pulses=1, pulse_sustain=480)
# subs_circuit_1.add_input(function='pwl', pulse_width=50, pulse_height=600,
#                        pulse_separation=10, number_of_pulses=1, pulse_sustain=480)

# POISSON SPIKES
subs_circuit.add_input(function='poisson-spikes', ts=np.arange(0.0, 100, step_size), dt=step_size, i_max=1e6, rate=0.2)
subs_circuit_1.add_input(function='poisson-spikes', ts=np.arange(0.0, 100, step_size), dt=step_size, i_max=1e6, rate=0.2)

subs_circuit.build_circuit(subcircuits, subcircuit_fnames, port_list=port_list, input_name=input_names, output_value=1,
                           outputs=output_list, step_size=step_size, parameters=param_dict)
subs_circuit_1.build_circuit(subcircuits_1, subcircuit_fnames, port_list=port_list, input_name=input_names, output_value=1,
                           outputs=output_list, step_size=step_size, parameters=param_dict_1)

# simulate new circuit
subs_circuit.run_simulation()
subs_circuit_1.run_simulation()
custom_labels = ['Input current (A)', 'Output Current (A)']
subs_circuit.plot_output(plot_idxs=[1, 3], subplot_idxs=[0, 1], save_fig=False, custom_labels=custom_labels,
                         compare_fname=out_fname_1)
# subs_circuit.plot_output(plot_idxs=[1, 4, 2], subplot_idxs=[0, 1, 2], save_fig=False,
#                          custom_labels=['Input Current (A)', 'JJ Voltage (mV)', 'JJ Phase'],
#                          scaling=[1, 1e3, 1])
subs_circuit.clear_output()
subs_circuit_1.clear_output()

inv_synapse_weights = np.arange(0.025, 1.025, 0.025)
synapse_scales = [str(np.round(1/inv_synapse_weights[k], 4)) for k in range(inv_synapse_weights.shape[0])]
# synapse_scales = ['5', '10', '15', '20', '25', '30', '35', '40', '45', '50']
# synapse_scale_array = np.arange(5, 55, 5)
synapse_scale_array = 1/inv_synapse_weights
amps = np.zeros(synapse_scale_array.shape[0])
for i in range(len(synapse_scales)):
    param_dict = dict()
    param_keys = ['parameters', 'model_names', 'types', 'values', 'idx']
    component_1exc = ['RSCALE', None, 'default', synapse_scales[i], 1]
    add_component(param_dict, component_1exc, param_keys)


    # build the circuit using build_circuit
    subs_circuit = Circuit(cir_fname=cir_fname, out_fname=out_fname, plot_fname=plot_fname, from_file=False,
                           fade_out_time=150)
    # input to emit single spike
    subs_circuit.add_input(function='pwl', pulse_width=50, pulse_height=600,
                           pulse_separation=10, number_of_pulses=1, pulse_sustain=480)

    subs_circuit.build_circuit(subcircuits, subcircuit_fnames, port_list=port_list, input_name=input_names,
                               output_value=1,
                               outputs=output_list, step_size=step_size, parameters=param_dict)
    subs_circuit.run_simulation()
    voltage = subs_circuit.get_time_series(idxs=3)
    amp = np.max(voltage)
    amps[i] = amp

import matplotlib.pyplot as plt
plt.plot(1/synapse_scale_array, amps*1e6)
plt.scatter(1/synapse_scale_array, amps*1e6, marker='x', c='k')
plt.xlabel('1/R_Scale (1/Ohm)')
plt.ylabel('Spike peak (µV)')
plt.grid()
plt.show()


