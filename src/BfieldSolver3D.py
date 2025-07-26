import numpy as np
import torch
import copy
from tqdm.auto import tqdm

import finite_differences_3d as fd3d
from ProgressionTracking import ProgressionTracking

class BfieldSolver3D(ProgressionTracking):

    def __init__(self, B_mag, B_x_angle, B_z_angle, x_res, y_res, z_res, Chi,
                 progression_tracking=None, analytical_solution=False,
                 device=None):
        """
        Parameters
        ----------
        B_mag : float [T]
            Magnitude of the B field.
        B_x_angle : float [radians]
            Angle with respect to the positive x-axis.
        B_z_angle : float [radians]
            Angle with respect to the positive z-axis.
        x_res : int
            Resolution in the x-axis. Number of pixels along x.
        y_res : int
            Resolution in the y-axis. Number of pixels along y.
        z_res : int
            Resolution in the z-plane. Number of pixels along each direction.
        Chi : torch.Tensor [-]
            The distribution of Chi for the substrate of interest.
        track_metrics : ...
            ...
        analytical_solution : ...
            ...
        """

        #### Device
        self.device = device

        #### B-field specs
        self.B_mag = B_mag
        self.B_x_angle = B_x_angle
        self.B_z_angle = B_z_angle
        self.x_res = x_res
        self.y_res = y_res
        self.z_res = z_res

        #### Substrate
        self.Chi = Chi.to(self.device)

        #### Initial Phi
        self.Phi, self.Phi_x_comp, self.Phi_y_comp, self.Phi_z_comp = self._initialize_phi()
        self.Phi = self.Phi.to(self.device)

        #### B-field
        self.B = self.compute_B_from_Phi()
        self.analytical_solution = analytical_solution

        #### ProgressionTracking
        # Inherit all functions from ProgressionTracking, and initialize all
        # variables from __init__(). Variables with same name in parent and sub
        # class will be overwritten if defined in sub class (here) are calling
        # super().__init__().

        self.progression_tracking = progression_tracking # TODO: Make this smarter

        if self.progression_tracking != None:

            super().__init__(**self.progression_tracking)

    def _initialize_phi(self):
        """
        Initializes the magentic scalar potential Phi.

        Returns
        -------
        Phi : torch.Tensor []
            The magnetic scalar potential.
        """

        precision = 1e-10

        z_comp  = np.cos(self.B_z_angle + np.pi * 2)
        if z_comp < precision: z_comp = 0.0

        xy_comp = np.sin(self.B_z_angle + np.pi * 2)

        x_comp = xy_comp * np.cos(self.B_x_angle + np.pi * 2)
        if x_comp < precision: x_comp = 0.0

        y_comp = xy_comp * np.sin(self.B_x_angle + np.pi * 2)
        if y_comp < precision: y_comp = 0.0

        x = np.arange(self.x_res) * x_comp * self.B_mag
        y = np.arange(self.y_res) * y_comp * self.B_mag
        z = np.arange(self.z_res) * z_comp * self.B_mag

        X,Y,Z = np.meshgrid(y, x, z)

        X = X[:self.x_res, :self.y_res, :self.z_res]
        Y = Y[:self.x_res, :self.y_res, :self.z_res]
        Z = Z[:self.x_res, :self.y_res, :self.z_res]

        Phi = X + Y + Z

        Phi = torch.from_numpy(Phi)
        print('Phi.shape: ', Phi.shape)

        #### Components
        Phi_x_comp = x_comp * self.B_mag
        Phi_y_comp = y_comp * self.B_mag
        Phi_z_comp = z_comp * self.B_mag

        return Phi, Phi_x_comp, Phi_y_comp, Phi_z_comp

    def compute_B_from_Phi(self):
        """
        TBW
        """

        l_x_corrected = (self.Phi[:, :1, :] + self.Phi_y_comp*(self.y_res))
        f_x_corrected = (self.Phi[:, -1:, :] - self.Phi_y_comp*(self.y_res))
        Phi_temp = torch.cat((f_x_corrected, self.Phi, l_x_corrected), dim=1)
        B_x = fd3d.dx_centered(Phi_temp, mode='no correction')

        l_y_corrected = (self.Phi[:1, :, :] + self.Phi_x_comp*(self.x_res))
        f_y_corrected = (self.Phi[-1:, :, :] - self.Phi_x_comp*(self.x_res))
        Phi_temp = torch.cat((f_y_corrected, self.Phi, l_y_corrected), dim=0)
        B_y = fd3d.dy_centered(Phi_temp, mode='no correction')

        l_z_corrected = (self.Phi[:, :, :1] + self.Phi_z_comp*(self.z_res))
        f_z_corrected = (self.Phi[:, :, -1:] - self.Phi_z_comp*(self.z_res))
        Phi_temp = torch.cat((f_z_corrected, self.Phi, l_z_corrected), dim=2)
        B_z = fd3d.dz_centered(Phi_temp, mode='no correction')

        print('self.Chi.shape', self.Chi.shape)
        print('B_x.shape', B_x.shape)
        print('B_y.shape', B_y.shape)
        print('B_z.shape', B_z.shape)
        print('B_mag.shape', self.B_mag.shape)

        self.B = (self.Chi + 1) * (1 - 2 / 3 * (self.Chi + 1 - 1)) * (+1) * \
                 torch.sqrt(B_x**2 + B_y**2 + B_z**2) - self.B_mag

        del Phi_temp, l_x_corrected, f_x_corrected, l_y_corrected, f_y_corrected, l_z_corrected, f_z_corrected
        torch.cuda.empty_cache()

        return self.B

    def solve_Phi_rough(self, max_iter, dT):
        """
        As described in C. M. Collins et al. ish. Results are very rippled.
        Horrible for the purpose, since gradients in the resulting field is
        crucial for DWI simulations.

        The methods allows for a larger dT, and thereby faster progression than
        solve_Phi_fine().

        Parameters
        ----------
        max_iter : int
            ...
        dT : float
            ...
        """

        self.dT = dT

        mu_r = self.Chi + 1.

        mu_r_dx = fd3d.dx_centered(mu_r, mode='roll')
        mu_r_dy = fd3d.dy_centered(mu_r, mode='roll')
        mu_r_dz = fd3d.dz_centered(mu_r, mode='roll')

        for i in tqdm(range(max_iter)):

            #### written explicitly
            l_x_corrected = (self.Phi[:, :2, :] + self.Phi_y_comp*(self.y_res))
            f_x_corrected = (self.Phi[:, -2:, :] - self.Phi_y_comp*(self.y_res))
            Phi_temp = torch.cat((f_x_corrected, self.Phi, l_x_corrected), dim=1)
            Phi_dx = fd3d.dx_centered(Phi_temp, mode='no correction')
            Phi_dxdx = fd3d.dx_centered(Phi_dx, mode='no correction')

            l_y_corrected = (self.Phi[:2, :, :] + self.Phi_x_comp*(self.x_res))
            f_y_corrected = (self.Phi[-2:, :, :] - self.Phi_x_comp*(self.x_res))
            Phi_temp = torch.cat((f_y_corrected, self.Phi, l_y_corrected), dim=0)
            Phi_dy = fd3d.dy_centered(Phi_temp, mode='no correction')
            Phi_dydy = fd3d.dy_centered(Phi_dy, mode='no correction')

            l_z_corrected = (self.Phi[:, :, :2] + self.Phi_z_comp*(self.z_res))
            f_z_corrected = (self.Phi[:, :, -2:] - self.Phi_z_comp*(self.z_res))
            Phi_temp = torch.cat((f_z_corrected, self.Phi, l_z_corrected), dim=2)
            Phi_dz = fd3d.dz_centered(Phi_temp, mode='no correction')
            Phi_dzdz = fd3d.dz_centered(Phi_dz, mode='no correction')

            Phi_x = mu_r_dx * Phi_dx[:, 1:-1, :] + mu_r * Phi_dxdx
            Phi_y = mu_r_dy * Phi_dy[1:-1, :, :] + mu_r * Phi_dydy
            Phi_z = mu_r_dz * Phi_dz[:, :, 1:-1] + mu_r * Phi_dzdz

            step = self.dT * (Phi_x + Phi_y + Phi_z)

            self.Phi_next = self.Phi + step

            if (self.progression_tracking != None) and (i % self.tracking_frequency == 0):
                self.update_tracked_metrics()

                l = len(self.metrics['summed_abs_change']) #to make it work when len()=1 as well
                if self.metrics['summed_abs_change'][l-1] > self.metrics['summed_abs_change'][l-2]:
                    print(f'\tSolving was terminated after {i} iterations because summed_abs_change was increasing.')
                    break

            self.Phi = copy.copy(self.Phi_next)

        #### clean up
        del self.dT

        torch.cuda.empty_cache()

        return self.Phi

    def solve_Phi_fine(self, max_iter, dT):
        """
        As described in C. M. Collins et al., but double derivatives are computed
        differently.

        The methods allows for a smaller dT, and thereby slower progression than
        solve_Phi_rough(). But the provides smooth (unrippled) results.

        Parameters
        ----------
        max_iter : int
            ...
        dT : float
            ...
        """

        self.dT = dT

        mu_r = self.Chi + 1.

        mu_r_dx = fd3d.dx_centered(mu_r, mode='roll')
        mu_r_dy = fd3d.dy_centered(mu_r, mode='roll')
        mu_r_dz = fd3d.dz_centered(mu_r, mode='roll')

        for i in tqdm(range(max_iter)):

            l_x_corrected = (self.Phi[:, :1, :] + self.Phi_y_comp*(self.y_res))
            f_x_corrected = (self.Phi[:, -1:, :] - self.Phi_y_comp*(self.y_res))
            Phi_temp = torch.cat((f_x_corrected, self.Phi, l_x_corrected), dim=1)
            Phi_dx = fd3d.dx_centered(Phi_temp, mode='no correction')
            Phi_dxdx = fd3d.dxdx_centered(Phi_temp, mode='no correction')

            l_y_corrected = (self.Phi[:1, :, :] + self.Phi_x_comp*(self.x_res))
            f_y_corrected = (self.Phi[-1:, :, :] - self.Phi_x_comp*(self.x_res))
            Phi_temp = torch.cat((f_y_corrected, self.Phi, l_y_corrected), dim=0)
            Phi_dy = fd3d.dy_centered(Phi_temp, mode='no correction')
            Phi_dydy = fd3d.dydy_centered(Phi_temp, mode='no correction')

            l_z_corrected = (self.Phi[:, :, :1] + self.Phi_z_comp*(self.z_res))
            f_z_corrected = (self.Phi[:, :, -1:] - self.Phi_z_comp*(self.z_res))
            Phi_temp = torch.cat((f_z_corrected, self.Phi, l_z_corrected), dim=2)
            Phi_dz = fd3d.dz_centered(Phi_temp, mode='no correction')
            Phi_dzdz = fd3d.dzdz_centered(Phi_temp, mode='no correction')

            Phi_x = mu_r_dx * Phi_dx + mu_r * Phi_dxdx
            Phi_y = mu_r_dy * Phi_dy + mu_r * Phi_dydy
            Phi_z = mu_r_dz * Phi_dz + mu_r * Phi_dzdz

            step = dT * (Phi_x + Phi_y + Phi_z)

            self.Phi_next = self.Phi + step

            if (self.progression_tracking != None) and (i % self.tracking_frequency == 0):
                self.update_tracked_metrics()

                l = len(self.metrics['summed_abs_change']) #to make it work when len()=1 as well
                if (self.metrics['dT'][l-1] == self.metrics['dT'][l-2]) and \
                   (self.metrics['summed_abs_change'][l-1] > self.metrics['summed_abs_change'][l-2]):
                    print(f'\tSolving was terminated after {i} iterations because summed_abs_change was increasing.')
                    break

            self.Phi = copy.copy(self.Phi_next)

        del self.dT

        return self.Phi
