import os
import numpy as np
from Circuit import Circuit
from Utils import add_component

out_fname = os.path.join('csv', 'temp.csv')

# create circuit from library of subcircuits
# define a new circuit from subcircuits and give it a new name
cir_name = f'temp.cir'
cir_fname = os.path.join('netlists', cir_name)
plot_fname = os.path.join('images', cir_name[:-4])

# setting up circuit details
# here the DC-DC-Converter circuit is build from its subcircuits
subcircuits = ["Neuron-Soma", "R-Synapse-exc"]
subcircuit_fnames = ['netlists/Neuron_circuits.cir']*len(subcircuits)
port_list = [[0, 1, 2], [0, 2, 3]]
output_list = ['DEVI IIN1', 'PHASE B0.X0', 'PHASE B1.X0', 'DEVV B0.X0', 'DEVV B1.X0', 'DEVI ROUT']
input_names = ['IIN1']

# write parameter dict to change subcircuits
param_dict = dict()
param_keys = ['parameters', 'model_names', 'types', 'values', 'idx']
component_1exc = ['RSCALE', None, 'default', '20', 1]

add_component(param_dict, component_1exc, param_keys)

# build the circuit using build_circuit
subs_circuit = Circuit(cir_fname=cir_fname, out_fname=out_fname, plot_fname=plot_fname, from_file=False,
                       fade_out_time=20, plot_latex=True)

step_size = 0.001

# POISSON SPIKES
subs_circuit.add_input(function='poisson-spikes', ts=np.arange(0.0, 300, step_size), dt=step_size, i_max=1e6, rate=0.5)

subs_circuit.build_circuit(subcircuits, subcircuit_fnames, port_list=port_list, input_name=input_names, output_value=1,
                           outputs=output_list, step_size=step_size, parameters=param_dict)


# simulate new circuit
subs_circuit.run_simulation()
custom_labels = ['Input current (A)', 'Output Current (A)']
subs_circuit.plot_label_size = 25
subs_circuit.plot_output(plot_idxs=[1], subplot_idxs=[0], save_fig=True, custom_labels=custom_labels, size=(7, 3),
                         y_labels=['Current in A'], grid=True, fname='z_input')
subs_circuit.plot_output(plot_idxs=[3], subplot_idxs=[0], save_fig=True, custom_labels=['Phase B1'],
                         y_labels=['Phase'], grid=True, size=(7, 3), fname='z_phase')
subs_circuit.plot_output(plot_idxs=[5], subplot_idxs=[0], save_fig=True, grid=True, size=(7, 3),
                         custom_labels=['Voltage B1'], y_labels=['v in mV'], scaling=[1e3],
                         fname='z_voltage')

subs_circuit.load_verification()
subs_circuit.get_rate(size=(7, 3), plot=True, calc_isi=True, plot_isi=True, show_cv=False, idxs=0, save_fig=True)



