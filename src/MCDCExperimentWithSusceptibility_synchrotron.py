import numpy as np
import torch
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class MCDCExperimentWithSusceptibility():

    def __init__(self, path_MCDC_config_file=None, device='cpu',
                 path_scheme_file=None):

        self.device = device

        torch.set_default_dtype(torch.float64)

        self._init_from_config_file(path_MCDC_config_file)

        if path_scheme_file != None:
            self.path_scheme_file = path_scheme_file

        #### some useful paths
        self.path_trajectories = '/'.join(self.path_pattern_out.split('/')[:-1])
        self.rep_tag = self.path_pattern_out.split('/')[-1]

        #### signals computed by MCDC
        self.signals_MCDC = np.fromfile(self.path_pattern_out+'_DWI.bfloat', dtype=np.float32)

        #### signals
        self.signals = {}

        self.dT = self.duration / self.T

        #### scheme file
        scheme = np.loadtxt(self.path_scheme_file, skiprows=1)
        scheme = torch.from_numpy(scheme).to(self.device)
        self.b_vectors = scheme[:, :3]
        self.b_vectors_unique = torch.unique(self.b_vectors, dim=0)
        self.Gs = scheme[:, 3]
        self.Gs_unique = torch.unique(self.Gs)
        self.Deltas = scheme[:, 4]
        self.deltas = scheme[:, 5]
        self.TEs = scheme[:, 6]
        self.TE_scheme = scheme[0, 6]
        self.T_scheme = torch.round(self.TE_scheme / self.dT).type(torch.int)

        self.gamma = torch.Tensor([267.513 * 10**6])[0].to(self.device) # [rad/(sT)] # gyromagnetic ratio for Hydrogen # 42.58 * 10**6 [Hz/T]
        self.b_values = (self.gamma * self.Gs * self.deltas)**2 * (self.Deltas - self.deltas/3)
        self.b_values_unique = torch.unique(self.b_values)

        ####
        self.n_dimensions = 3

        ####
        # self.PAs = self.get_powder_average()

    def _init_from_config_file(self, path_MCDC_config_file):
        '''
        TBW.

        '''

        with open(path_MCDC_config_file, 'r') as file:

            for line in file:

                if 'N ' in line:
                    self.N = int(line.strip().split(' ')[-1])
                elif 'T ' in line:
                    self.T = int(line.strip().split(' ')[-1])
                elif 'duration ' in line:
                    self.duration = float(line.strip().split(' ')[-1])
                elif 'diffusivity' in line:
                    self.diffusivity = float(line.strip().split(' ')[-1])
                elif 'scheme_file' in line:
                    self.path_scheme_file = str(line.strip().split(' ')[-1])
                elif 'out_traj_file_index' in line:
                    self.path_pattern_out = str(line.strip().split(' ')[-1])
                elif 'cylinders_list' in line:
                    self.path_cylinders_list = str(line.strip().split(' ')[-1])
                elif 'ply ' in line:
                    self.path_ply = str(line.strip().split(' ')[-1])
                elif '<voxels>' in line:
                    voxel_min = next(file)
                    voxel_max = next(file)
                    self.voxel_xmin, self.voxel_ymin, self.voxel_zmin = np.array(voxel_min.strip().split(' ')).astype(float)
                    self.voxel_xmax, self.voxel_ymax, self.voxel_zmax = np.array(voxel_max.strip().split(' ')).astype(float)

    def load_trajectories(self, names_trajectories_files):
        """
        TBW

        Parameters
        ----------
        ...

        Returns
        -------
        trajectories : torch.Tensor [self.N, self.T+1, self.n_dimensions]
            ...
        """

        trajectories = np.array([])

        #for filename_oi in tqdm(names_trajectories_files):
        for filename_oi in names_trajectories_files:
            trajectories_tmp = np.fromfile(os.path.join(self.path_trajectories, filename_oi), dtype=np.float32)
            trajectories = np.append(trajectories, trajectories_tmp)

        n_points = int(trajectories.shape[0] / self.n_dimensions)

        N_here = int(trajectories.shape[0] / self.n_dimensions / (self.T+1))

        trajectories = np.reshape(trajectories, (N_here, self.T+1, self.n_dimensions))
        trajectories = torch.from_numpy(trajectories).to(self.device)

        trajectories *= 1e-3

        return trajectories

    def delete_trajectories(self):
        """
        Proper clearing from GPU.

        """

        del self.trajectories
        torch.cuda.empty_cache()

    def compute_gradient_contribution(self, trajectories):
        """
        inspired by: https://github.com/jonhrafe/MCDC_Simulator_public/blob/master/src/pgsesequence.cpp

        This:
            torch.einsum('nwc,c->nw', trajectories[:, firstBlockStart:firstBlockEnd+1, :]* 1e-3, b_vector)
        Corresponds to:
            for i in range(firstBlockStart, firstBlockEnd):
                pos_current_all = trajectories[:, i, :] * 1e-3

        """

        pads = (self.TE_scheme - self.Deltas - self.deltas) / 2.0

        firstBlockStarts = (pads / self.dT).type(torch.int)
        firstBlockEnds = ((pads + self.deltas) / self.dT).type(torch.int)

        secondBlockStarts = ((pads + self.Deltas) / self.dT).type(torch.int)
        secondBlockEnds = ((pads + self.Deltas + self.deltas) / self.dT).type(torch.int)

        gradient_contribs = []

        for b_vector, G, firstBlockStart, firstBlockEnd, secondBlockStart, secondBlockEnd in zip(self.b_vectors, self.Gs, firstBlockStarts, firstBlockEnds, secondBlockStarts, secondBlockEnds):

            # gradient contribution from block 1
            gc_1 = torch.einsum('nwc,c->nw', trajectories[:, firstBlockStart:firstBlockEnd+1, :], b_vector)
            gc_1 = torch.sum(gc_1, dim=-1)

            # gradient contribution from block 2
            gc_2 = torch.einsum('nwc,c->nw', trajectories[:, secondBlockStart:secondBlockEnd+1, :], b_vector)
            gc_2 = torch.sum(gc_2, dim=-1)

            gradient_contrib = (gc_1 - gc_2) * G

            gradient_contribs.append(gradient_contrib)

        gradient_contribs = torch.stack(gradient_contribs)

        return gradient_contribs

    def compute_susceptibility_contribution(self, trajectories, paths_Bfields, debug=False, check_rogue=False, label_oi=None, path_segmentation=None):

        verbose = False

        # terrible solution
        if check_rogue:
            if label_oi == 2:
                label_no_go = 1
            elif label_oi == 0:
                label_no_go = 1

        if check_rogue:
            print('path_segmentation', path_segmentation)
            segmentation = torch.load(path_segmentation).to(self.device)
            print('segmentation.shape', segmentation.shape)
            if verbose: print('path_segmentation', path_segmentation)
            if verbose: print('segmentation', torch.unique(segmentation))

            if verbose: print('segmentation.shape: ', segmentation.shape)

        susceptibility_contribs = []
        mask_rogue_particles = []

        if verbose: print('self.N: ', self.N)

        for path_Bfield in np.sort(paths_Bfields):

            # initialize susceptibility_contrib as zeros.
            susceptibility_contrib = torch.zeros((self.N)).type(torch.float64).to(self.device)

            # load Bfield and send to device
            B = torch.load(path_Bfield).to(self.device)
            if check_rogue:
                assert segmentation.shape == B.shape

            if verbose: print('B.shape', B.shape)

            n_x, n_y, n_z = B.shape

            res_x = float(path_Bfield.split('-res_actual_x=')[-1].split('-')[0])
            res_y = float(path_Bfield.split('-res_actual_y=')[-1].split('-')[0])
            res_z = float(path_Bfield.split('-res_actual_z=')[-1].split('-')[0].replace('.pt', ''))

            len_x = n_x * res_x * 1e-6
            len_y = n_y * res_y * 1e-6
            len_z = n_z * res_z * 1e-6

            if verbose: print('len_x: ', len_x)
            if verbose: print('len_y: ', len_y)
            if verbose: print('len_z: ', len_z)

            if verbose: print('trajectories: ', trajectories.shape)

            if verbose:
                plt.figure(figsize=(8, 8))
                for traj in trajectories[:5]:
                    plt.scatter(traj[:, 0].to('cpu'), traj[:, 1].to('cpu'), alpha=0.05)
                plt.axis('equal')
                plt.grid(True)
                plt.title('0')
                plt.show()

            if verbose: print('self.voxel_xmin', self.voxel_xmin * 1e-3)
            if verbose: print('self.voxel_xmax', self.voxel_xmax * 1e-3)
            if verbose: print('self.voxel_ymin', self.voxel_ymin * 1e-3)
            if verbose: print('self.voxel_ymax', self.voxel_ymax * 1e-3)
            if verbose: print('self.voxel_zmin', self.voxel_zmin * 1e-3)
            if verbose: print('self.voxel_zmax', self.voxel_zmax * 1e-3)
            
            #### susceptibility contribution from block 1
            pos_1 = trajectories[:, :self.T_scheme//2, :].clone().detach()

            if verbose:
                plt.figure(figsize=(8, 8))
                for pos in pos_1[:5]:
                    plt.scatter(pos[:, 0].to('cpu'), pos[:, 1].to('cpu'), alpha=0.05)
                plt.axis('equal')
                plt.grid(True)
                plt.title('1')
                plt.show()

            if verbose: print('traj', torch.min(pos_1[:, :, 0]), torch.max(pos_1[:, :, 0]))
            if verbose: print('traj', torch.min(pos_1[:, :, 1]), torch.max(pos_1[:, :, 1]))
            if verbose: print('traj', torch.min(pos_1[:, :, 2]), torch.max(pos_1[:, :, 2]))

            # shift apparent voxel edge to 0
            pos_1[:, :, 0] += (-1)*self.voxel_xmin * 1e-3
            pos_1[:, :, 1] += (-1)*self.voxel_ymin * 1e-3
            pos_1[:, :, 2] += (-1)*self.voxel_zmin * 1e-3

            if verbose:
                plt.figure(figsize=(8, 8))
                for pos in pos_1:
                    plt.scatter(pos[:, 0].to('cpu'), pos[:, 1].to('cpu'), alpha=0.05)
                plt.axis('equal')
                plt.grid(True)
                plt.title('2 - yes')
                plt.show()

            if verbose: print('traj', torch.min(pos_1[:, :, 0]), torch.max(pos_1[:, :, 0]))
            if verbose: print('traj', torch.min(pos_1[:, :, 1]), torch.max(pos_1[:, :, 1]))
            if verbose: print('traj', torch.min(pos_1[:, :, 2]), torch.max(pos_1[:, :, 2]))

            pos_1[:, :, 0] = torch.round(pos_1[:, :, 0] / len_x * (n_x - 1)).type(torch.int) #% n_x
            pos_1[:, :, 1] = torch.round(pos_1[:, :, 1] / len_y * (n_y - 1)).type(torch.int) #% n_y
            pos_1[:, :, 2] = torch.round(pos_1[:, :, 2] / len_z * (n_z - 1)).type(torch.int) #% n_z

            # because torch.Tensor([13]) % 3 != torch.Tensor([-13]) % 3
            pos_1[:, :, 0][pos_1[:, :, 0] > 0.] = pos_1[:, :, 0][pos_1[:, :, 0] > 0.] % (n_x)
            pos_1[:, :, 0][pos_1[:, :, 0] < 0.] = pos_1[:, :, 0][pos_1[:, :, 0] < 0.] % (-n_x)
            pos_1[:, :, 1][pos_1[:, :, 1] > 0.] = pos_1[:, :, 1][pos_1[:, :, 1] > 0.] % (n_y)
            pos_1[:, :, 1][pos_1[:, :, 1] < 0.] = pos_1[:, :, 1][pos_1[:, :, 1] < 0.] % (-n_y)
            pos_1[:, :, 2][pos_1[:, :, 2] > 0.] = pos_1[:, :, 2][pos_1[:, :, 2] > 0.] % (n_z)
            pos_1[:, :, 2][pos_1[:, :, 2] < 0.] = pos_1[:, :, 2][pos_1[:, :, 2] < 0.] % (-n_z)

            pos_1[:, :, 0][pos_1[:, :, 0] < 0.] += n_x
            pos_1[:, :, 1][pos_1[:, :, 1] < 0.] += n_y
            pos_1[:, :, 2][pos_1[:, :, 2] < 0.] += n_z

            idxs_1 = pos_1.clone().detach().type(torch.long)
            if verbose: print('idxs_1: ', idxs_1.shape)

            if verbose:
                plt.figure(figsize=(10, 10))

                for i_row in range(segmentation.shape[0]):
                    for i_col in range(segmentation.shape[1]):

                        if segmentation[i_row, i_col, 0].to('cpu') == 0:
                            color='blue'
                        elif segmentation[i_row, i_col, 0].to('cpu') == 1:
                            color='green'
                        elif segmentation[i_row, i_col, 0].to('cpu') == 2:
                            color='yellow'
                        plt.scatter(i_row, i_col, c=color, s=80)

                for pos in idxs_1:
                    plt.scatter(pos[:, 0].to('cpu'), pos[:, 1].to('cpu'),
                                alpha=0.1,
                                c='black', marker='x'
                                )

                plt.axis('equal')
                plt.grid(True)
                plt.title('3')
                plt.show()

            if check_rogue:
                a, b = torch.unique(segmentation[torch.flatten(idxs_1[:, :, 0]), torch.flatten(idxs_1[:, :, 1]), torch.flatten(idxs_1[:, :, 2])], return_counts=True)
                some = torch.reshape(segmentation[torch.flatten(idxs_1[:, :, 0]), torch.flatten(idxs_1[:, :, 1]), torch.flatten(idxs_1[:, :, 2])], idxs_1[:, :, 2].shape)

                mask_rogue_particles_1 = torch.sum(some == label_no_go, dim=1) == 0

            sc_1 = torch.sum(torch.reshape(B[torch.flatten(idxs_1[:, :, 0]), torch.flatten(idxs_1[:, :, 1]), torch.flatten(idxs_1[:, :, 2])], idxs_1[:, :, 2].shape), dim=-1)

            if verbose: print('sc_1.shape: ', sc_1.shape)
            if verbose: print('torch.flatten(idxs_1[:, :, 0]): ', torch.flatten(idxs_1[:, :, 0]).shape)
            if verbose: print('idxs_1[:, :, 0]: ', idxs_1[:, :, 0].shape)

            #### susceptibility contribution from block 2
            pos_2 = trajectories[:, self.T_scheme//2:self.T_scheme, :].clone().detach()

            # shift apparent voxel edge to 0
            pos_2[:, :, 0] += (-1)*self.voxel_xmin * 1e-3
            pos_2[:, :, 1] += (-1)*self.voxel_ymin * 1e-3
            pos_2[:, :, 2] += (-1)*self.voxel_zmin * 1e-3

            pos_2[:, :, 0] = torch.round(pos_2[:, :, 0] / len_x * (n_x - 1)).type(torch.int) #% n_x
            pos_2[:, :, 1] = torch.round(pos_2[:, :, 1] / len_y * (n_y - 1)).type(torch.int) #% n_y
            pos_2[:, :, 2] = torch.round(pos_2[:, :, 2] / len_z * (n_z - 1)).type(torch.int) #% n_z

            pos_2[:, :, 0][pos_2[:, :, 0] > 0.] = pos_2[:, :, 0][pos_2[:, :, 0] > 0.] % (n_x)
            pos_2[:, :, 0][pos_2[:, :, 0] < 0.] = pos_2[:, :, 0][pos_2[:, :, 0] < 0.] % (-n_x)
            pos_2[:, :, 1][pos_2[:, :, 1] > 0.] = pos_2[:, :, 1][pos_2[:, :, 1] > 0.] % (n_y)
            pos_2[:, :, 1][pos_2[:, :, 1] < 0.] = pos_2[:, :, 1][pos_2[:, :, 1] < 0.] % (-n_y)
            pos_2[:, :, 2][pos_2[:, :, 2] > 0.] = pos_2[:, :, 2][pos_2[:, :, 2] > 0.] % (n_z)
            pos_2[:, :, 2][pos_2[:, :, 2] < 0.] = pos_2[:, :, 2][pos_2[:, :, 2] < 0.] % (-n_z)

            pos_2[:, :, 0][pos_2[:, :, 0] < 0.] += n_x
            pos_2[:, :, 1][pos_2[:, :, 1] < 0.] += n_y
            pos_2[:, :, 2][pos_2[:, :, 2] < 0.] += n_z

            idxs_2 = pos_2.clone().detach().type(torch.long)
            if check_rogue:

                print('idxs_2: ', idxs_2.shape)
                
                a, b = torch.unique(segmentation[torch.flatten(idxs_2[:, :, 0]), torch.flatten(idxs_2[:, :, 1]), torch.flatten(idxs_2[:, :, 2])], return_counts=True)
                
                print('unique_2: ', a, b)
                
                some = torch.reshape(segmentation[torch.flatten(idxs_2[:, :, 0]), torch.flatten(idxs_2[:, :, 1]), torch.flatten(idxs_2[:, :, 2])], idxs_2[:, :, 2].shape)

                mask_rogue_particles_2 = torch.sum(some == label_no_go, dim=1) == 0

            sc_2 = torch.sum(torch.reshape(B[torch.flatten(idxs_2[:, :, 0]), torch.flatten(idxs_2[:, :, 1]), torch.flatten(idxs_2[:, :, 2])], idxs_2[:, :, 2].shape), dim=-1)

            #### combine
            susceptibility_contrib = sc_1 - sc_2

            # susceptibility_contribs.append(susceptibility_contrib[mask_rogue_particles])
            if check_rogue: mask_rogue_particles.append(~torch.logical_or(~mask_rogue_particles_1, ~mask_rogue_particles_2))
            susceptibility_contribs.append(susceptibility_contrib)

        del pos_1, idxs_1, pos_2, idxs_2
        torch.cuda.empty_cache()

        susceptibility_contribs = torch.stack(susceptibility_contribs)
        if check_rogue: mask_rogue_particles = torch.stack(mask_rogue_particles)

        if debug:
            return susceptibility_contribs, map
        else:
            return susceptibility_contribs, mask_rogue_particles

    def get_powder_average(self):
        '''
        TBW
        '''

        PAs = []

        for b_value_oi in self.b_values_unique:

            signals_oi = self.signals[self.b_values == b_value_oi]

            PAs.append(np.mean(signals_oi))

        return PAs

    def load_cylinders_list(self):

        with open(self.path_cylinders_list) as f:

            for line in f:
                scale_cylinders_list = float(line)
                break

        cylinders_list = np.atleast_2d(np.loadtxt(self.path_cylinders_list, skiprows=1, delimiter=' '))

        cylinders_list = cylinders_list * scale_cylinders_list * 1e-3

        return cylinders_list, scale_cylinders_list

    def get_compartment_masks(self, trajectories):

        # load cylinder list
        cylinders_list, _ = self.load_cylinders_list()
        cylinders_list = torch.Tensor(cylinders_list).to(self.device)

        pbar = tqdm(zip(cylinders_list[::2], cylinders_list[1::2]), total=len(cylinders_list[::2]))
        pbar.set_description('Cylinders in substrate')

        mask_intra = torch.zeros((len(trajectories))).bool().to(self.device)

        for cylinder_o, cylinder_i in pbar:

            center_x, center_y, _, _, _, _, r_o = cylinder_o
            _, _, _, _, _, _, r_i = cylinder_i

            xs = trajectories[:, 0] - center_x
            ys = trajectories[:, 1] - center_y

            # check if intra
            mask_intra += xs**2 + ys**2 < r_i**2

            # check if myelin
            # not relevant

            # check if extra
            #mask_extra = xs**2 + ys**2 > r_o

        mask_extra = torch.logical_not(mask_intra)

        return mask_intra, mask_extra

    def fit_DTI(self, bvals, bvecs, volume, big_delta=None, small_delta=None, b0_threshold=None,
                b_thres=None, min_signal=None):

        import dipy.reconst.dti as dti
        import dipy.data
        from dipy.reconst.dti import fractional_anisotropy, color_fa

        #### gtab
        b_mask = bvals < b_thres
        gtab = dipy.data.gradient_table(bvals[b_mask], bvecs=bvecs[b_mask, :],
                                        big_delta=big_delta, small_delta=small_delta,
                                        b0_threshold=b0_threshold)

        #### Instantiate the Tensor model
        tenmodel = dti.TensorModel(gtab, return_S0_hat=True, min_signal=min_signal)

        #### Fit the data
        tenfit = tenmodel.fit(volume[:, :, :, b_mask])

        self.tenfit = tenfit

        evals = tenfit.evals[:,:,np.newaxis,:]
        evecs = tenfit.evecs[:,:,np.newaxis,:,:]

        FA = fractional_anisotropy(tenfit.evals)

        #In the background of the image the fitting will not be accurate there is no signal and possibly we will find FA
        #values with nans (not a number). We can easily remove these in the following way.
        FA[np.isnan(FA)] = 0
        FA = np.clip(FA, 0, 1)

        #get color FA
        RGB = color_fa(FA, tenfit.evecs)

        #We can color the ellipsoids using the color_fa values that we calculated above. In this example we additionally
        #normalize the values to increase the contrast.
        cfa = RGB[:,:,np.newaxis,:]
        cfa /= cfa.max()

        #Other tensor statistics can be calculated from the tenfit object. For example, a commonly calculated statistic is
        #the mean diffusivity (MD). This is simply the mean of the eigenvalues of the tensor. Since FA is a normalized
        #measure of variance and MD is the mean, they are often used as complimentary measures. In DIPY, there are two
        #equivalent ways to calculate the mean diffusivity. One is by calling the mean_diffusivity module function on the
        #eigen-values of the TensorFit class instance:
        MD = dti.mean_diffusivity(tenfit.evals)

        self.evals = evals
        self.eval1 = evals[:,:,0,:,0]
        self.eval2 = evals[:,:,0,:,1]
        self.eval3 = evals[:,:,0,:,2]
        self.evecs = evecs
        self.FA = FA
        self.MD = MD
        self.RGB = RGB
        self.cfa = cfa

        self.measurements = {'evals': evals,'eval1': self.eval1,'eval2': self.eval2,'eval3': self.eval3,'evecs':evecs,'FA':FA,'MD':MD,'RGG':RGB,'cfa':cfa}#,'ROIs_TensorModel':ROIs_TensorModel}

        return tenfit
