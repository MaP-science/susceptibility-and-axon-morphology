# sys.argv[1] = 'name_substrate_type'

#### ENVIRONMENT ###############################################################

import os
import sys
import numpy as np
import torch
import pytorch3d
import pytorch3d.io
from tqdm.auto import tqdm

torch.set_printoptions(16)

sys.path.append('../../')
import MCDC_analysis.src.GenerateMCDCConfigFile as mcdc_config
import MCDC_analysis.src.initial_positions_generation as ipg
from MCDC_substrate_making.src.CylinderListEnsembleGenerator import CylinderListEnsembleGenerator
import MCDC_substrate_making.src.cylinder_list_segmentation_generation as clsg
from ISMRM2020.src.PathPolice_ISMRM2020 import PathPolice_ISMRM2020 as PathPolice
import NumB.src.compute_Bfields_from_paths_segmentations as compute_Bfields
import mesh_utils.src.mesh_io as mesh_io

#### SET NAMES AND PATHS #######################################################

keys_unwanted = ['_outer']

path_home = '/dtu-compute/siwin/'

name_project = 'susceptibility_in_silico'
name_substrate_type = sys.argv[1]

path_police = PathPolice(path_home, name_project, '', name_substrate_type)

#### MCDC path specs

name_experiment = 'microdispersion'
path_pattern_out = f'{path_police.path_project}/data/MCDC/{name_experiment}/'
name_scheme = 'n_shells=05-n_directions=3' #'n_shells=09-n_directions=21'
path_scheme_file = f'{path_police.path_project}/resources/DWI/schemes/{name_scheme}.scheme'

if not os.path.exists(path_pattern_out):

    os.system(f'mkdir -p {path_pattern_out}')

#### mesh paths oi

# has n_voxels_required < 65000000 for buffer = 5*0.0655 for G6s.
white_list = ['axon06', 'axon08', 'axon12', 'axon13', 'axon14',
              'axon15', 'axon17', 'axon18', 'axon22', 'axon24', 'axon25',
              'axon26', 'axon27', 'axon28', 'axon31', 'axon32', 'axon33',
              'axon34', 'axon38', 'axon40', 'axon41', 'axon43', 'axon45',
              'axon46', 'axon47', 'axon48', 'axon49', 'axon50', 'axon51',
              'axon52', 'axon53', 'axon54']
# black_list = ['axon10', ]

path_data = os.path.join(path_home, 'projects/susceptibility_in_silico/substrates/')
path_meshes = os.path.join(path_data, name_substrate_type, 'meshes')
path_segmentations = os.path.join(path_data, name_substrate_type, 'segmentations')
names_segmentations = os.listdir(path_segmentations)


paths_meshes_outer = np.sort([os.path.join(path_meshes, n) for n in os.listdir(path_meshes) if ('outer.ply' in n) and any([tag in n for tag in white_list])])
paths_meshes_inner = np.sort([os.path.join(path_meshes, n) for n in os.listdir(path_meshes) if ('inner.ply' in n) and any([tag in n for tag in white_list])])

### voxel specs
res_desired = 0.0655 #0.0645 #0.0655 #0.1 #0.064 # [um]

buffer_x = 5 * res_desired
buffer_y = 5 * res_desired
# if ('G1' in name_substrate_type) or ('G3' in name_substrate_type):
#     buffer_z = 0.0
# else:
#     buffer_z = 5 * res_desired
if ('G1' in name_substrate_type):
    buffer_x = 35 * res_desired
    buffer_y = 35 * res_desired
    buffer_z = -2280 * res_desired
elif ('G3' in name_substrate_type):
    buffer_x = 35 * res_desired
    buffer_y = 35 * res_desired
    buffer_z = 5 * res_desired
else:
    buffer_z = 5 * res_desired


#### MCDC CONFIG SPECS #########################################################

# number of walkers to include in ini_walker_pos-file
# number of time steps
T = 10000#57000 #10000
# duration in seconds
duration = 0.036 #Delta+delta
# diffusivity
diffusivity = 0.6e-9#0.6e-9#2e-9 #### 18-04-2021: 2e-9 CORRESPONDS TO IN VIVO. 0.6e-9 CORRESPONDS TO EX VIVO.
# number of processors to use
num_process = 16#16

if ('G1' in name_substrate_type):
    buffer_sampling_area = [0.0, 0.0, 0.00005] # [mm] #0.00005
    density_particles = 1500 # [1/um^3]
else:
    buffer_sampling_area = [0.0, 0.0, 0.020] # [mm] #0.00005
    density_particles = 15 # [1/um^3]
# buffer_sampling_area = [0.0, 0.0, 0.050] # [mm] #0.00005
print('\n\n\n buffer_sampling_area = [0.0, 0.0, 0.050] \n\n\n')

print('\nThese parameters result in:')
dT = duration / T
print('\t dT: ', dT)
len_step = np.sqrt(6 * diffusivity * dT)
print('\t len_step: ', len_step)
# input('Continue?\n')

#### GET MESH PATHS OI #############################################################

for path_mesh_inner, path_mesh_outer in zip(tqdm(paths_meshes_inner), paths_meshes_outer):

    print('path_mesh_inner: ', path_mesh_inner)

    #### GENERATE SEGMENTATIONS ################################################
    #### load meshes
    verts_outer, faces_outer = mesh_io.load_ply(path_mesh_outer)

    #### get voxel specs
    voxel_xmin = verts_outer[:, 0].min() - buffer_x
    voxel_xmax = verts_outer[:, 0].max() + buffer_x
    voxel_ymin = verts_outer[:, 1].min() - buffer_y
    voxel_ymax = verts_outer[:, 1].max() + buffer_y
    voxel_zmin = verts_outer[:, 2].min() - buffer_z
    voxel_zmax = verts_outer[:, 2].max() + buffer_z

    voxel_lims = [voxel_xmin, voxel_xmax, voxel_ymin, voxel_ymax, voxel_zmin, voxel_zmax]

    #### get intra-volume
    tag_axon = path_mesh_inner.split('/')[-1].split('-')[0]
    mask = [(tag_axon in n) and (f'res_desired={res_desired}' in n) for n in names_segmentations]
    assert np.sum(mask) <= 1, f'{names_segmentations} \n {mask}'
    if np.sum(mask) < 1:
        print('no segment for this one... will therefore be skipped...')
        continue
    idx_oi = np.argmax(mask)
    path_segmentation = os.path.join(path_segmentations, names_segmentations[idx_oi])
    res_actual_x = float(path_segmentation.split('res_actual_x=')[-1].split('-')[0])
    res_actual_y = float(path_segmentation.split('res_actual_y=')[-1].split('-')[0])
    res_actual_z = float(path_segmentation.split('res_actual_z=')[-1].split('-')[0].replace('.pt', ''))
    segmentation = torch.load(path_segmentation)
    volume_intra = torch.sum(segmentation == 2) * (res_actual_x * res_actual_y * res_actual_z)
    N_particles = int(np.ceil(volume_intra * density_particles))
    print(f'N_particles = {N_particles}')

    #### generate config-file
    path_config_file = mcdc_config.write_config_file(path_mesh_inner, path_pattern_out,
                                                     density_particles,
                                                     T, duration, diffusivity,
                                                     num_process, path_scheme_file,
                                                     buffer_sampling_area,
                                                     voxel_lims=voxel_lims,
                                                     N_particles=N_particles)

    #### RUN MCDC-SIMULATION
    #if os.path.exists(path_config_file.replace('.conf', '_DWI.bfloat')):
    #    print(f'{path_config_file} was already run... will therefore be skipped...')
    if path_config_file == None:
        continue
    else:
        print(f'path_config_file:\n\t {path_config_file}')
        os.system(f'MC-DC_Simulator {path_config_file}')
