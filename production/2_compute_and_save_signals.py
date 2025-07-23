import os
import sys

# sys.argv[1] = 'idx_gpu'
# sys.argv[2] = 'name_substrate_type'
# sys.argv[3] = 'check_rogue'
# sys.argv[4] = 'tag_signals'
# sys.argv[5] = 'tag_scheme'
# sys.argv[6] = 'B_x_angle'

# python 3_compute_and_save_signals.py 7 synchrotron-G6-z_aligned True -max_iter_rough=50000-max_iter_fine=15000 -Delta=20.2-delta=7.2 0.0
# python 3_compute_and_save_signals.py 7 hexagonal_helix_undulated-n_trims=1-MWF-r_tube_outer=1.50 False -max_iter_rough=50000-max_iter_fine=15000 -Delta=20.2-delta=7.2 0.0
# python 3_compute_and_save_signals.py 0 TEST=0.85-r_tube_outer=1.50 True -max_iter_rough=50000-max_iter_fine=15000 -Delta=20.2-delta=7.2 0.0

try:
    B_x_angle = sys.argv[6]
except:
    B_x_angle = '0.00'

#### ENVIRONMENT ###############################################################

# Make GPU visible. Must be done before torch is imported.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] #index of gpu of interest #provide as command line input

import numpy as np
import torch
import sys
from tqdm.auto import tqdm
import pickle

sys.path.append('../../')
# from MCDC_analysis.src.MCDCExperimentWithSusceptibility import MCDCExperimentWithSusceptibility
from MCDC_analysis.src.MCDCExperimentWithSusceptibility_synchrotron import MCDCExperimentWithSusceptibility

# Check available devices. Choose GPU if one is available. Else choose cpu.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f'number of devices: {torch.cuda.device_count()}')
print(f'device: {device}')

# allocate GPU
dummy = torch.Tensor([0.0]).to(device)

#### KEYS FOR FILTERING ########################################################

keys_oi_config = [] ####
keys_oi_Bfield = [f'B_x_angle={B_x_angle}']
n_Bfields_required = 7

check_rogue = sys.argv[3] == 'True'

#### PATHS #####################################################################

# tag_signals = '-max_iter_rough=500-max_iter_fine=1500'
# tag_signals = '-max_iter_rough=15000-max_iter_fine=5000'
# tag_signals = '-max_iter_rough=50000-max_iter_fine=15000'
# tag_signals = '-max_iter_rough=50000-max_iter_fine=150000'
# tag_signals = '-max_iter_rough=75000-max_iter_fine=5000'
tag_signals = sys.argv[4]
tag_scheme = sys.argv[5]

#### home dir
path_home = '/dtu-compute/siwin/'

####
name_project = 'susceptibility_in_silico'
path_project = os.path.join(path_home, 'projects', name_project)

#### MCDC path specs
name_experiment = 'microdispersion'
path_pattern_out = f'{path_project}/data/MCDC/{name_experiment}'

path_substrates = os.path.join(path_home, 'projects', name_project, 'substrates')

name_substrate_type = sys.argv[2]#['synchrotron-G1-z_aligned']#, 'synchrotron-G3-z_aligned', 'synchrotron-G4-z_aligned', 'synchrotron-G5-z_aligned', 'synchrotron-G6-z_aligned']

# name_scheme_file = 'n_shells=9-n_directions=21-TE=0.036.scheme'
name_scheme_file = 'n_shells=7-n_directions=21-TE=0.036-delta=0.0072-Delta=0.0202.scheme'

path_scheme_file = os.path.join(path_project, 'resources', 'DWI', 'schemes', name_scheme_file)

path_data = os.path.join(path_pattern_out, name_substrate_type)
path_Bfields = os.path.join(path_project, 'substrates', name_substrate_type, 'Bfields'+tag_signals) ####
path_segmentations = os.path.join(path_project, 'substrates', name_substrate_type, 'segmentations')
names_configs = [name for name in os.listdir(path_data) if '.conf' in name]
paths_configs = np.sort([os.path.join(path_data, name) for name in names_configs if all([key in name for key in keys_oi_config])])

#### ACTION ####################################################################

for path_config in tqdm(paths_configs):

    print('path_config: ', path_config)

    name_config = path_config.split('/')[-1].replace('.conf', '')
    name_scheme = name_scheme_file.replace('.scheme', '')
    #name_signal_file = f'{name_config}-{name_scheme}{tag_signals}.signals' ####
    if B_x_angle == '0.00':
        name_signal_file = f'{name_config}-{name_scheme}{tag_signals}{tag_scheme}.signals' ####
    else:
        name_signal_file = f'{name_config}-{name_scheme}{tag_signals}{tag_scheme}-B_x_angle={B_x_angle}.signals' ####
    path_signals = os.path.join(path_data, name_signal_file)

    if os.path.exists(path_signals):
        print(f'{path_signals} was skipped because it already exists...\n')
        continue

    if not os.path.exists(path_config.replace('.conf', '_DWI.bfloat')):
        print(f'{path_signals} was skipped because MCDC did not yet finish...\n')
        continue

    experiment = MCDCExperimentWithSusceptibility(path_MCDC_config_file=path_config,
                                                  device=device,
                                                  path_scheme_file=path_scheme_file,)

    if 'synchrotron' in path_config:
        tag_axon = path_config.split('/')[-1].split('-')[0]
    elif 'helix' in path_config:
        tag_axon = '-'.join(path_config.split('/')[-1].split('-')[:6])

    # get the paths of the associated B-fields
    names_Bfields = [name for name in os.listdir(path_Bfields) if (tag_axon in name) and ('-log' not in name) and all([key in name for key in keys_oi_Bfield])]

    paths_Bfields = np.sort([os.path.join(path_Bfields, name_Bfield) for name_Bfield in names_Bfields])
    if len(paths_Bfields) < n_Bfields_required:
        print(f'{path_signals} was skipped because len(paths_Bfields) < n_Bfields_required...\n')
        continue

    path_segmentation = paths_Bfields[0].replace(tag_signals, '').replace('Bfields', 'segmentations').split('-B_x_angle')[0] + '.pt' ####
    ####path_segmentation = path_segmentation.replace('-inner-', '-')

    names_trajectories_files = [f for f in os.listdir(experiment.path_trajectories) if (experiment.rep_tag in f) and ('.traj' in f)]

    signals_gradient = torch.zeros((len(experiment.b_values))).to(device)
    signals_susceptibility = torch.zeros((len(paths_Bfields), len(experiment.b_values))).to(device)
    signals_net = torch.zeros((len(paths_Bfields), len(experiment.b_values))).to(device)

    for name_trajectories_file in tqdm(names_trajectories_files):

        #### load trajectories
        trajectories = experiment.load_trajectories([name_trajectories_file])#.type(torch.float64)

        #### compute susceptibility contribution
        if check_rogue:

            if 'inner' in path_config:
                label_oi = 2
            elif 'outer' in path_config:
                label_oi = 0

            susceptibility_contribs, mask_rogue_particles = experiment.compute_susceptibility_contribution(
                trajectories,
                paths_Bfields,
                check_rogue=True,
                path_segmentation=path_segmentation,
                label_oi=label_oi)

            print('mask_rogue_particles')
            print(mask_rogue_particles.shape)
            print(torch.sum(mask_rogue_particles, dim=1))
            print('trajectories', trajectories.shape)
            _n_particles_in_trajectories = trajectories.shape[0]
            _n_rogue_particles = _n_particles_in_trajectories - torch.sum(mask_rogue_particles, dim=1)[0]
            print('number of rogue particles in .traj file: ', _n_rogue_particles)
            print('fraction of rogue particles in .traj file: ', _n_rogue_particles/_n_particles_in_trajectories)
            print()
        else:
            susceptibility_contribs, _ = experiment.compute_susceptibility_contribution(trajectories,
                                                                                        paths_Bfields,)

        #### compute gradient contribution
        gradient_contribs = experiment.compute_gradient_contribution(trajectories)

        if check_rogue:
            #### rogue masking
            susceptibility_contribs = susceptibility_contribs[:, mask_rogue_particles[0, :]]
            gradient_contribs = gradient_contribs[:, mask_rogue_particles[0, :]]

        # compute signals
        phases_gradient = gradient_contribs * experiment.gamma * experiment.dT
        signals_gradient += torch.sum(torch.cos(phases_gradient), dim=-1)
        phases_susceptibility = susceptibility_contribs * experiment.gamma * experiment.dT
        signals_susceptibility += torch.sum(torch.cos(phases_susceptibility.unsqueeze(1).repeat((1, phases_gradient.shape[0], 1))), dim=-1)
        signals_net += torch.sum(torch.cos(phases_susceptibility.unsqueeze(1).repeat((1, phases_gradient.shape[0], 1)) + phases_gradient), dim=-1)

    # add single instance of gradient contribution
    experiment.signals['only_diffusion_gradients'] = signals_gradient
    # add dictionary of contributions. one for each of the supplied Bfields.
    experiment.signals['only_susceptibility_gradients'] = {path_Bfield : signal for path_Bfield, signal in zip(paths_Bfields, signals_susceptibility)}
    experiment.signals['all_gradients'] = {path_Bfield : signal for path_Bfield, signal in zip(paths_Bfields, signals_net)}

    experiment.signals['path_scheme_file'] = path_scheme_file

    #### save
    with open(path_signals, 'wb') as file:
        pickle.dump(experiment.signals, file)
