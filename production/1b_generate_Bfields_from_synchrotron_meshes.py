
verbose = False

#### EXPECTED COMMAND LINE ARGUMENTS ###########################################

# sys.argv[1] = 'idx_gpu'
# sys.argv[2] = 'name_subsrate_type'
# sys.argv[3] = 'max_iter_rough' #50000 #15000
# sys.argv[4] = 'max_iter_fine' #150000 #5000
# sys.argv[5] = 'order'
# sys.argv[6] = 'off_set'

#### ENVIRONMENT ###############################################################

import os
import sys

# Make GPU visible. Must be done before torch is imported.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] #index of gpu of interest #provide as command line input

# import
import numpy as np
import torch

sys.path.append('../src')
from PathPolice_validate import PathPolice_validate as PathPolice
import compute_Bfields_from_paths_segmentations as compute_Bfields

# Check available devices. Choose GPU if one is available. Else choose cpu.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f'number of devices: {torch.cuda.device_count()}')
print(f'device: {device}')

# allocate GPU
dummy = torch.Tensor([0.0]).to(device)

#### PATHS #####################################################################

path_home = '../'

name_project = 'susceptibility_in_silico'
name_substrate_type = sys.argv[2]

#### SUBSTRATE SPECS ###########################################################

# has n_voxels_required < 65000000 for buffer = 5*0.0655 for G6s.
white_list = ['axon06', 'axon08', 'axon12', 'axon13', 'axon14',
              'axon15', 'axon17', 'axon18', 'axon22', 'axon24', 'axon25',
              'axon26', 'axon27', 'axon28', 'axon31', 'axon32', 'axon33',
              'axon34', 'axon38', 'axon40', 'axon41', 'axon43', 'axon45',
              'axon46', 'axon47', 'axon48', 'axon49', 'axon50', 'axon51',
              'axon52', 'axon53', 'axon54']
# black_list = ['axon10', ]

#### segmentation
# res_desired = 0.0122 # [um]
res_desired = 0.0655 #0.0645 #0.0655 #0.1 #0.064 # [um]

#### Bfield SPECS ##############################################################

class ConfigBfield():
    def __init__(self):
        pass

config_Bfield = ConfigBfield()

#### B-field
config_Bfield.B_mags = np.array([np.float64(7.0)]) # [T]
config_Bfield.B_x_angles = np.deg2rad(np.array([0])).astype(np.float64) # angle wrt. postive x-axis [radians]
config_Bfield.B_z_angles = np.deg2rad(np.array([0, 15, 30, 45, 60, 75, 90])).astype(np.float64) # angle wrt. postive z-axis [radians]

#### substrate
config_Bfield.label_i = 2
config_Bfield.label_m = 1
config_Bfield.label_o = 0

Chi_water = np.float64(0.99999096)-np.float64(1)
Chi_fat = np.float64(0.99999221)-np.float64(1)

MWF = 0.15 #MWF estimated from: https://www.frontiersin.org/articles/10.3389/fnins.2020.00136/full

config_Bfield.Chi_i = Chi_water #magnetic susceptibility inside
config_Bfield.Chi_o = Chi_water #magnetic susceptibility outside
config_Bfield.Chi_m = MWF*Chi_water + (1-MWF)*Chi_fat #magnetic susceptibility membrane

#### progression tracking
config_Bfield.progression_tracking = {'metric_names' : ['dT', 'summed_abs_change', ],
                                      'tracking_frequency' : 100,}

#### iterations
config_Bfield.max_iter_rough = int(sys.argv[3]) #15000#5000#500#500#50000
config_Bfield.dT_rough = 0.65
config_Bfield.max_iter_fine = int(sys.argv[4]) #5000#15000#1500#150000
config_Bfield.dT_fine = 0.16

tag_quick = f'-max_iter_rough={config_Bfield.max_iter_rough}-max_iter_fine={config_Bfield.max_iter_fine}'

#### GET MESHES OI #############################################################

keys_all = [f'res_desired={res_desired}'] # all these keys must be in str

path_data = os.path.join(path_home, 'projects/susceptibility_in_silico/substrates/')
path_meshes = os.path.join(path_data, name_substrate_type, 'meshes')
path_Bfields = os.path.join(path_data, name_substrate_type, 'Bfields'+tag_quick)
path_segmentations = os.path.join(path_data, name_substrate_type, 'segmentations')

#### COMPUTE B-FIELDS ##########################################################

paths_segmentations = [os.path.join(path_segmentations, n) for n in os.listdir(path_segmentations) if all(k in n for k in keys_all)]

for path_segmentation in paths_segmentations:

    print('\n    path_segmentation: ', path_segmentation)

    path_police = PathPolice(path_home, name_project, '', name_substrate_type)
    compute_Bfields.compute([path_segmentation], device, path_police, config_Bfield, quick=tag_quick)
