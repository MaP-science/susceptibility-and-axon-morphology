import os
import numpy as np


class PathPolice_validate():

    def __init__(self, path_home, name_project, name_dwi_scheme, name_substrate_type):

        self.path_home = path_home

        # project
        self.name_project = name_project
        self.path_project = os.path.join(self.path_home, 'projects', self.name_project)
        if not os.path.exists(self.path_project):
            os.makedirs(self.path_project)

        # figures
        self.path_figures = os.path.join(self.path_project, 'figures')
        if not os.path.exists(self.path_figures):
            os.makedirs(self.path_figures)

        # resources
        self.path_resources = os.path.join(self.path_project, 'resources')
        if not os.path.exists(self.path_resources):
            os.makedirs(self.path_resources)

        # DWI sequence scheme
        self.name_dwi_scheme = name_dwi_scheme

        # substrate
        self.name_substrate_type = name_substrate_type
        self.path_master_substrates = os.path.join(self.path_project, 'substrates') # path to all master substrates
        if not os.path.exists(self.path_master_substrates):
            os.makedirs(self.path_master_substrates)
        self.names_master_substrates = os.listdir(self.path_master_substrates) # paths to individual master substrates

        self.paths_individual_substrates = []
        for name_master_substrate in self.names_master_substrates:
            self.paths_individual_substrates += [os.path.join(self.path_master_substrates, name_master_substrate, file) for file in os.listdir(os.path.join(self.path_master_substrates, name_master_substrate)) if ('gamma' in file) and ('gamma_distributed_cylinder_list.txt' not in file)]

        self.paths_individual_substrates = np.sort(self.paths_individual_substrates)

    #### HERE
    def get_path_master_substrate_from_path_cylinder_list(self, path_cylinder_list):
        return '/'.join(path_cylinder_list.split('/')[:-1])

    def get_name_master_substrate_from_path_cylinder_list(self, path_cylinder_list):
        return path_cylinder_list.split('/')[-2]

    def get_path_segmentations_from_path_cylinder_list(self, path_cylinder_list):
        return '/'.join(path_cylinder_list.split('/')[:-1]) + '/segmentations'

    def get_name_segmentation_from_path_segmentation(self, path_segmentation):
        return path_segmentation.split('/')[-1]

    def get_path_Bfields_from_path_segmentation(self, path_segmentation):
        return '/'.join(path_segmentation.split('/')[:-2]) + '/Bfields'

    def get_path_simulation_info_file_from_path_cylinder_list(self, path_cylinder_list):
        return '/'.join(path_cylinder_list.split('/')[:-1]) + '/' + '_'.join(path_cylinder_list.split('/')[-1].split('_')[:2]) + '_simulation_info.txt'

    def get_g_ratio_from_path_cylinder_list(self, path_cylinder_list):
        return float(path_cylinder_list.split('/')[-1].split('g_ratio=')[-1].split('-')[0])

    def get_rep_tag_from_path_cylinder_list(self, path_cylinder_list):
        return '_'.join(path_cylinder_list.split('/')[-1].split('_')[:2])

    def get_full_tag_from_path_cylinder_list(self, path_cylinder_list):
        return path_cylinder_list.split('/')[-1].replace('.txt', '')

    #### DEP
    def path_master_substrate(self, name_master_substrate):
        return os.path.join(self.path_master_substrates, name_master_substrate)

    def path_sub_substrate(self, name_master_substrate, name_sub_substrate):
        return os.path.join(self.path_master_substrates, name_master_substrate, name_sub_substrate)

    def path_segmentations(self, name_master_substrate):
        return os.path.join(self.path_master_substrates, name_master_substrate, 'segmentations')

    def path_Bfields(self, name_substrate):
        return os.path.join(self.path_master_substrates, name_substrate, 'Bfields')

    def name_simulation_info_file(self, name_sub_substrate):
        return '_'.join(name_sub_substrate.split('_')[:2]) + '_simulation_info.txt'

    def rep_tag(self, name_sub_substrate):
        return '_'.join(name_sub_substrate.split('_')[:2])

    # def name_simulation_info_file(self, )

    def get_g_ratio_from_name_sub_substrate(self, name_sub_substrate):
        return float(name_sub_substrate.split('g_ratio=')[-1].split('-')[0])
