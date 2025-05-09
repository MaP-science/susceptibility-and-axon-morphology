import os
import sys

verbose = False

# sys.argv[1] = 'idx_gpu'
# sys.argv[2] = 'name_substrate_type'
# sys.argv[3] = 'order'

try:
    order = sys.argv[3]
except:
    order = 1

#### ENVIRONMENT ###############################################################

# Make GPU visible. Must be done before torch is imported.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] #index of gpu of interest #provide as command line input

# import
import numpy as np
import torch
import pytorch3d
import pytorch3d.io
import math
import matplotlib.pyplot as plt
import trimesh
from tqdm.auto import tqdm

sys.path.append('../../')
import plotting.src.utils.meshes as pm
import MCDC_substrate_making.src.plotting as plotting
import mesh_utils.src.close_roundish_holes as crh
import mesh_utils.src.mesh_io as mesh_io
import MCDC_substrate_making.src.building_blocks as bb
import MCDC_substrate_making.src.hexagonal_helixes as hh
from plotting.src.utils.slice_viewing import ImageSliceViewer3D
import mesh_utils.src.mesh_to_voxels as m2v
from MCDC_analysis.src.PathPolice_validate import PathPolice_validate as PathPolice
import NumB.src.compute_Bfields_from_paths_segmentations as compute_Bfields

# Check available devices. Choose GPU if one is available. Else choose cpu.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f'number of devices: {torch.cuda.device_count()}')
print(f'device: {device}')

# allocate GPU
dummy = torch.Tensor([0.0]).to(device)

#### PATHS #####################################################################

path_home = '/dtu-compute/siwin/'

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
#res_desired = 0.0655 #0.0645 #0.0655 #0.1 #0.064 # [um]
res_desired = 0.0655 #0.0645 #0.0655 #0.1 #0.064 # [um]

n_trims = 1 # number of rounds of triming of the myelin

####
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
####
# ####
# buffer_x = 0.0
# buffer_y = 0.0
# if ('G1' in name_substrate_type) or ('G3' in name_substrate_type):
#     buffer_z = 0.0
# else:
#     buffer_z = 0.0
# ####

thres_n_voxels = 65000000

#### GET MESHES OI #############################################################

keys_all = [f'res_desired={res_desired}'] # all these keys must be in str

path_data = os.path.join(path_home, 'projects/susceptibility_in_silico/substrates/')
path_meshes = os.path.join(path_data, name_substrate_type, 'meshes')
path_Bfields = os.path.join(path_data, name_substrate_type, 'Bfields')
# path_segmentations = os.path.join(path_data, name_substrate_type, 'segmentations')
path_segmentations = os.path.join(path_data, name_substrate_type, 'segmentations-res_desired=%f.4' %(res_desired))

paths_meshes_outer = np.sort([os.path.join(path_meshes, n) for n in os.listdir(path_meshes) if ('outer.ply' in n) and any([tag in n for tag in white_list])])
paths_meshes_inner = np.sort([os.path.join(path_meshes, n) for n in os.listdir(path_meshes) if ('inner.ply' in n) and any([tag in n for tag in white_list])])

#### GENERATE ALL ##############################################################

for path_mesh_inner, path_mesh_outer in zip(tqdm(paths_meshes_inner[::order]), paths_meshes_outer[::order]):

    print('path_mesh_inner: ', path_mesh_inner)

    #### GENERATE SEGMENTATIONS ################################################
    #### load meshes
    verts_inner, faces_inner = mesh_io.load_ply(path_mesh_inner)
    verts_outer, faces_outer = mesh_io.load_ply(path_mesh_outer)

    #### get voxel specs
    voxel_xmin = verts_outer[:, 0].min() - buffer_x
    voxel_xmax = verts_outer[:, 0].max() + buffer_x
    voxel_ymin = verts_outer[:, 1].min() - buffer_y
    voxel_ymax = verts_outer[:, 1].max() + buffer_y
    voxel_zmin = verts_outer[:, 2].min() - buffer_z
    voxel_zmax = verts_outer[:, 2].max() + buffer_z

    voxel_length_x = voxel_xmax - voxel_xmin
    voxel_length_y = voxel_ymax - voxel_ymin
    voxel_length_z = voxel_zmax - voxel_zmin

    if verbose: print('voxel_length_x, voxel_length_y, voxel_length_z: ', voxel_length_x, voxel_length_y, voxel_length_z)

    res_actual_x = voxel_length_x / np.round(voxel_length_x / res_desired)
    res_actual_y = voxel_length_y / np.round(voxel_length_y / res_desired)
    res_actual_z = voxel_length_z / np.round(voxel_length_z / res_desired)

    #### check voxel threshold
    n_voxels_required = (voxel_length_x / res_actual_x) * (voxel_length_y / res_actual_y) * (voxel_length_z / res_actual_z)

    if n_voxels_required > thres_n_voxels:
        print(f'skipped because n_voxels_required = {n_voxels_required}...')
        continue
    else:
        print(f'\n   n_voxels_required = {n_voxels_required}...\n')

    if verbose: print('res_desired', res_desired)
    if verbose: print('res_actual_x, res_actual_y, res_actual_z: ', res_actual_x, res_actual_y, res_actual_z)

    #### check if exists
    path_segmentation = path_mesh_inner.replace('meshes', 'segmentations').replace('.ply', f'-res_desired={res_desired:.4f}-res_actual_x={res_actual_x:.16f}-res_actual_y={res_actual_y:.16f}-res_actual_z={res_actual_z:.16f}.pt')

    if os.path.exists(path_segmentation):

        print(f'{path_segmentation} already exists... will therefore be skipped...')

    else:

        #### About the segmentation
        # The myelin compartment must be:
        #    - strictly OUTSIDE the INNER mesh
        #    - strictly INSIDE the OUTER mesh

        #### get individual segmentations
        # for inner:
        # a voxel classified as outside the inner mesh, must be completely outside the mesh.
        # True means classified as inside.
        segmentation_inner = m2v.get_segmentation_from_mesh(path_mesh_inner,
                                                            bbox=[voxel_xmin, voxel_xmax,
                                                                  voxel_ymin, voxel_ymax,
                                                                  voxel_zmin, voxel_zmax,],
                                                            res=[res_actual_x, res_actual_y, res_actual_z],
                                                            device=device,
                                                            mode='strictly outside',
                                                           )
        # for outer:
        # a voxel classified as inside the outer mesh, must be completely inside the outer mesh.
        # True means classified as inside
        segmentation_outer = m2v.get_segmentation_from_mesh(path_mesh_outer,
                                                            bbox=[voxel_xmin, voxel_xmax,
                                                                  voxel_ymin, voxel_ymax,
                                                                  voxel_zmin, voxel_zmax,],
                                                            res=[res_actual_x, res_actual_y, res_actual_z],
                                                            device=device,
                                                            mode='strictly inside',
                                                           )

        #### get net segmentation
        segmentation_net = ~segmentation_inner.type(torch.int) + segmentation_outer.type(torch.int) + 2

        #### trim segmentation
        for _ in tqdm(range(n_trims)):
            segmentation_net = m2v.trim_segmentation(segmentation_net)

        #### save segmentation
        torch.save(segmentation_net, path_segmentation)
