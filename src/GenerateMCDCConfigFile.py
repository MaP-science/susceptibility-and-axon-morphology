import os
import math
import numpy as np

class GenerateMCDCConfigFile():
    """
    #### specs of diffusion system
    N = 10000 # [#] # number of particles
    T = 5000 # [#] # number of timesteps
    duration = 0.050000000 # [s] # duration of the simulation
    diffusivity = 0.0000000006 # [m^2/s] # diffusion coefficient
    deportation = 1 # whether to discard particles after illegal crossings. Detecting numerical errors. Correcting. Only works for very well-defined structures.
    #sphere_size = #radius used for collision detection. tricky to tune. using default i convenient.

    #### specs of output files
    # Cons for binary files:
    #     - has much higher precision
    #     - takes up much less memory
    write_txt = 0 # whether to output .txt-files
    write_bin = 1 # whether to output binary files
    write_traj_file = 1 # whether to output trajectories

    #### scaling
    # scale_from_stu = 1 indicates that  units in config file are SI-units
    # scale_from_stu != 1 indicates that units in the config file should be scaled with this parameter to obtain SI-units.
    # [m], [s] will be scaled to [mm], [ms], and so on.
    # Affects schemefile, duration, diffusivity
    scale_from_stu = 1 # scaling w.r.t. standard units (SI units)

    #### resources
    num_process = 4 # [#] # number of processors to use for computation
    # num_process = -2, means to use all available processors except 2.
    #max_sim_time = 19800 # Run the simulation for as many particles possible within this time.

    #### console outputs
    verbatim = 0
    """

    def __init__(self, path_file, allow_overwrite=False, **kwargs):
        """

        """

        self.parameters_simple = kwargs['parameters_simple']
        self.parameters_complex = kwargs['parameters_complex']

        self.names_parameters_simple = []
        self.names_parameters_complex = []

        self.path_file = path_file

        #### create file
        if not os.path.exists(self.path_file):
            # os.makedirs(os.path.dirname(self.path_file), exist_ok=False)
            with open(self.path_file, 'w'):
                pass
        elif os.path.exists(self.path_file) and allow_overwrite == True:
            os.makedirs(os.path.dirname(self.path_file), exist_ok=True)
            with open(self.path_file, 'w'):
                pass
        else:
            print(f'{self.path_file} already exists and allow_overwrite = {allow_overwrite}...')

    def _write_content_to_file(self, content):

        with open(self.path_file, 'a') as f:
            f.write(content+'\n')

    def _write_complex(self, key, value):

        self._write_content_to_file(f'<{key}>')
        for key_, value_ in value.items():
            if type(value_) == dict:
                self._write_complex(key_, value_)
            elif type(value_) == list:
                for elem in value_:
                    content = f'{key_} {elem}'
                    self._write_content_to_file(content)
            else:
                content = f'{key_} {value_}'
                self._write_content_to_file(content)
        self._write_content_to_file(f'</{key}>')

    def write_file(self):

        #### write simple parameters
        for key, value in self.parameters_simple.items():
            content = f'{key} {value}'
            self._write_content_to_file(content)

        self._write_content_to_file('')

        #### write complex parameters
        for key, value in self.parameters_complex.items():
            self._write_complex(key, value)
            self._write_content_to_file('')

        self._write_content_to_file('<END>')


def write_config_file(path_mesh, path_pattern_out, density_particles, T,
                      duration, diffusivity, num_process, path_scheme_file,
                      buffer_sampling_area, scale=1e-3, voxel_lims=None,
                      N_particles=None):

    if type(buffer_sampling_area) != list:
        buffer_sampling_area = [buffer_sampling_area, buffer_sampling_area, buffer_sampling_area]

    name_substrate_master = path_mesh.split('/')[-3]

    # path_pattern_out_oi = path_pattern_out + name_substrate_master + '/'
    path_pattern_out_oi = os.path.join(path_pattern_out, name_substrate_master)
    # if not os.path.exists(path_pattern_out_oi):
    #     os.system(f'mkdir -p {path_pattern_out_oi}')
    if not os.path.exists(path_pattern_out_oi):
        os.makedirs(path_pattern_out_oi)

    tag_phantom = path_mesh.split('/')[-1].replace('.ply', '')

    path_config_file = os.path.join(path_pattern_out_oi, tag_phantom)+'.conf'#f'{path_pattern_out_oi}{ply_tag}.conf'

    if os.path.exists(path_config_file):
        print(f'{path_config_file} already exists and will therefore be skipped...')
        return None

    if voxel_lims == None:
        #### get voxel specs from .ply-header
        with open(path_mesh, 'r') as file:
            for line in file:
                if 'voxel_xmin' in line:
                    voxel_xmin = float(line.strip().split('voxel_xmin=')[-1])
                if 'voxel_ymin' in line:
                    voxel_ymin = float(line.strip().split('voxel_ymin=')[-1])
                if 'voxel_zmin' in line:
                    voxel_zmin = float(line.strip().split('voxel_zmin=')[-1])
                if 'voxel_xmax' in line:
                    voxel_xmax = float(line.strip().split('voxel_xmax=')[-1])
                if 'voxel_ymax' in line:
                    voxel_ymax = float(line.strip().split('voxel_ymax=')[-1])
                if 'voxel_zmax' in line:
                    voxel_zmax = float(line.strip().split('voxel_zmax=')[-1])
    else:
        voxel_xmin, voxel_xmax, voxel_ymin, voxel_ymax, voxel_zmin, voxel_zmax = voxel_lims


    # compute volume of unit cell
    voxel_length_x = voxel_xmax - voxel_xmin
    voxel_length_y = voxel_ymax - voxel_ymin
    voxel_length_z = voxel_zmax - voxel_zmin
    voxel_volume = voxel_length_x * voxel_length_y * voxel_length_z

    if 'axon' in path_mesh.lower():
        compartment = 'intra'
        N_particles = N_particles
    else:
        #icvf = float(path_mesh.split('icvf=')[-1].split('-')[0])
        ecvf = float(path_mesh.split('ecvf=')[-1].split('-')[0])
        icvf = 1 - ecvf
        g_ratio = float(path_mesh.split('g_ratio=')[-1].split('-')[0])
        r_tube_outer = float(path_mesh.split('r_tube_outer=')[-1].split('/')[0])

        if '-outer.ply' in path_mesh:
            compartment = 'extra'
            # print('extra volume', voxel_volume * (1.0 - icvf))
            N_particles = int(np.ceil(voxel_volume * (1.0 - icvf) * density_particles))
        elif '-inner.ply' in path_mesh:
            compartment = 'intra'
            icvf_corrected = math.pi * (r_tube_outer * g_ratio)**2 * 2 * voxel_length_z / voxel_volume # because icvf in path is actually also myelin compartment. FIX THAT!
            # mvf = (math.pi * (r_tube_outer)**2 - math.pi * (r_tube_outer * g_ratio)**2) * 2 * voxel_length_z # for checking
            # print('intra volume', voxel_volume * icvf_corrected)
            N_particles = int(np.ceil(voxel_volume * icvf_corrected * density_particles))
            # N_particles = int(np.ceil(voxel_volume / (scale**3) * icvf_corrected * density_particles))

    #### parameters

    # simple
    parameters_simple = {'N' : N_particles,
                         'T' : T,
                         'duration' : duration, #Delta+delta
                         'diffusivity' : diffusivity, #3e-9, #2e-5,
                         'deportation' : 1,
                         ####'deportation' : 0,
                         'write_txt' : 0,
                         'write_bin' : 1,
                         'write_traj_file' : 1,
                         'scale_from_stu' : 1,
                         'num_process' : num_process,
                         'verbatim' : 0,
                         'scheme_file' : path_scheme_file,
                         'ini_walker_pos' : compartment,
                         'out_traj_file_index' : os.path.join(path_pattern_out_oi, tag_phantom),}

    # complex
    voxel_min_str = f'%.16f %.16f %.16f' %(voxel_xmin*scale, voxel_ymin*scale, voxel_zmin*scale)
    voxel_max_str = f'%.16f %.16f %.16f' %(voxel_xmax*scale, voxel_ymax*scale, voxel_zmax*scale)
    voxels = f'{voxel_min_str} \n {voxel_max_str}'

    sampling_area_min_str = f'%.16f %.16f %.16f' %(voxel_xmin*scale+buffer_sampling_area[0], voxel_ymin*scale+buffer_sampling_area[1], voxel_zmin*scale+buffer_sampling_area[2])
    sampling_area_max_str = f'%.16f %.16f %.16f' %(voxel_xmax*scale-buffer_sampling_area[0], voxel_ymax*scale-buffer_sampling_area[1], voxel_zmax*scale-buffer_sampling_area[2])
    sampling_area = f'{sampling_area_min_str} \n {sampling_area_max_str}'

    # obstacle
    obstacle = {'ply' : path_mesh,
                'ply_scale' : scale,}#1.0,}

    parameters_complex = {'obstacle' : obstacle,
                          'voxels' : {'' : voxels,},
                          'sampling_area' : {'' : sampling_area,}}

    parameters = {'parameters_simple' : parameters_simple,
                  'parameters_complex' : parameters_complex,}

    #### generate config file
    config_file_generator = GenerateMCDCConfigFile(path_config_file, allow_overwrite=True, **parameters)

    config_file_generator.write_file()

    return path_config_file
