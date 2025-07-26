from tqdm.auto import tqdm
import torch
import os
import numpy as np
import json

from BfieldSolver3D import BfieldSolver3D



def compute(paths_segmentations, device, pp, config, boundary_mode='circular', quick=False):

    for i, path_segmentation in tqdm(enumerate(paths_segmentations)):

        name_segmentation = pp.get_name_segmentation_from_path_segmentation(path_segmentation)
        print('name_segmentation: ', name_segmentation)
        path_Bfields = pp.get_path_Bfields_from_path_segmentation(path_segmentation)
        if quick != False:
            path_Bfields = path_Bfields.replace('Bfields', 'Bfields'+quick)
        print('path_Bfields: ', path_Bfields)
        if not os.path.exists(path_Bfields):
            os.makedirs(path_Bfields)

        #### load segmentation
        segmentation = torch.load(path_segmentation).type(torch.DoubleTensor)

        if segmentation.shape[-1] == 1:
            segmentation = segmentation.repeat(1, 1, 3)

        #### generate substrate
        x_res, y_res, z_res = np.shape(segmentation) 

        print('x_res, y_res, z_res: ', x_res, y_res, z_res)

        substrate_Chi = torch.zeros(np.shape(segmentation)).type(torch.DoubleTensor)

        substrate_Chi[segmentation == config.label_i] = config.Chi_i
        substrate_Chi[segmentation == config.label_m] = config.Chi_m
        substrate_Chi[segmentation == config.label_o] = config.Chi_o

        for B_mag in config.B_mags:

            print(f'\tB_mag = \t{B_mag}')

            for B_x_angle in config.B_x_angles:

                print(f'\tB_x_angle = \t{B_x_angle}')

                for B_z_angle in config.B_z_angles: ####

                    print(f'\tB_z_angle = \t{B_z_angle}')

                    name_Bfield = name_segmentation.replace('.pt', '-B_x_angle=%.2f-B_z_angle=%.2f.pt' %(B_x_angle, B_z_angle))
                    path_Bfield = path_Bfields+'/'+name_Bfield

                    if os.path.exists(path_Bfield):
                        print(f'\n    Bfield already exists: {path_Bfield}')

                        continue

                    #### initialize solver
                    BSolver = BfieldSolver3D(B_mag, B_x_angle, B_z_angle, x_res, y_res, z_res, substrate_Chi,
                                             progression_tracking=config.progression_tracking,
                                             device=device,)

                    #### solve Phi and compute B(Phi)

                    # rough method (using only first order derivatives)
                    BSolver.solve_Phi_rough(max_iter=config.max_iter_rough, dT=config.dT_rough)

                    del BSolver.Phi_next
                    torch.cuda.empty_cache()

                    #### inspect memory consumption
                    # check_memory_consumption()

                    # fine method (using also second order derivatives)
                    Phi_fine = BSolver.solve_Phi_fine(max_iter=config.max_iter_fine, dT=config.dT_fine)
                    B_fine = BSolver.compute_B_from_Phi()

                    #### save B-field
                    # Bfield
                    torch.save(B_fine.cpu(), path_Bfield)

                    # progression metrics
                    name_Bfield_log = name_Bfield.replace('.pt', '-log.pt')
                    with open(os.path.join(path_Bfields, name_Bfield_log), 'w') as file:
                        json.dump(BSolver.metrics, file)

                    del Phi_fine, B_fine
                    del BSolver.Phi, BSolver.Phi_next, BSolver.B, BSolver.Chi
                    torch.cuda.empty_cache()
