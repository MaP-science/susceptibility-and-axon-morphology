import numpy as np
import nibabel as nib

class ScanData(object):

    def __init__(self, path_images, path_bs_eff, path_mask_ROIs, labels_ROIs, tag=None, marker='.'):

        self.tag = tag
        self.marker = marker
        
        if ('M0686-N' in path_images) or ('M0687-N' in path_images) or ('M0688-N' in path_images) or ('M0689-N' in path_images):
            a = np.array([50, 4000, 8000, 12000, 20000,]) #desired b-values
        elif ('M0694-N' in path_images) or ('M0695-N' in path_images) or ('M0697-N' in path_images) or ('M0698-N' in path_images):
            a = np.array([50, 1000, 2000, 3000, 4000,]) #desired b-values
        elif ('M0714-N' in path_images) or ('M0715-N' in path_images) or ('M0716-N' in path_images) or ('M0717-N' in path_images):
            a = np.array([50, 1000, 3000, 4000, 8000, 12000, 20000]) #desired b-values

        #Load scan data
        V = nib.load(path_images)
        self.V = V.get_fdata()
        self.affine = V.affine
        pixdim = V.header['pixdim']
        srow_x = V.header['srow_x']
        srow_y = V.header['srow_y']
        srow_z = V.header['srow_z']

        self.M = np.vstack((srow_x[:3], srow_y[:3], srow_z[:3])) #rotation matrix

        normalizer = np.matlib.repmat(pixdim[1:4], 3, 1)
        self.M = self.M / normalizer

        self.bvals_eff = np.atleast_2d(np.loadtxt(path_bs_eff, skiprows=0)[:,-1]).T
        
        # get desired b-values
        self.bvals = a[np.argmin(np.abs(self.bvals_eff - a), axis=-1)][:, np.newaxis]
        self.bvals_unique = np.unique(self.bvals)

        self.bvecs_eff = np.loadtxt(path_bs_eff, skiprows=0)[:,:3]
        self.bvecs = np.copy(self.bvecs_eff)
        _, idxs_unique = np.unique(self.bvecs, axis=0, return_index=True)
        self.bvecs_unique = self.bvecs[np.sort(idxs_unique), :] #to keep the order
        
        self._average_over_even_bvals()

        #### load masks
        if path_mask_ROIs != None:
            self.mask_ROIs = nib.load(path_mask_ROIs).get_fdata()
            self.labels_ROIs = labels_ROIs
        
    def _average_over_even_bvals(self):
        
        V_new = []
        bvals_eff_new = []
        bvals_new = []

        bvecs_eff_new = []
        bvecs_new = []

        for bvec in self.bvecs_unique:

            if np.linalg.norm(bvec) == 0.0:
                print('WARNING: bvec=[0.0, 0.0, 0.0] is disregarded and discarded!')
                continue
            
            for bval in self.bvals_unique:

                if bval == 0.0:
                    print('WARNING: b=0.0 is disregarded and discarded!')
                    continue

                idxs_bvecs_oi = np.argwhere(self.bvecs == bvec)[:, 0][::3]
                idxs_bvals_oi = np.argwhere(self.bvals == bval)[:, 0]

                idxs_oi = np.intersect1d(idxs_bvecs_oi, idxs_bvals_oi)

                V_new.append(np.mean(self.V[:, :, :, idxs_oi], axis=-1))
                bvals_eff_new.append(self.bvals_eff[idxs_oi[0], :])
                bvals_new.append(self.bvals[idxs_oi[0], :])
                bvecs_eff_new.append(self.bvecs_eff[idxs_oi[0], :])
                bvecs_new.append(self.bvecs[idxs_oi[0], :])

                assert np.array_equal(bvec, self.bvecs[idxs_oi[0], :])

        self.V = np.transpose(np.array(V_new), [1, 2, 3, 0])
        self.bvals_eff = np.array(bvals_eff_new)
        self.bvals = np.array(bvals_new)

        self.bvecs_eff = np.array(bvecs_eff_new)
        self.bvecs = np.array(bvecs_new)
    
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
        
        
    
    def fit_DKI(self, bvals, bvecs, volume, big_delta=None, small_delta=None, b0_threshold=None, 
                b_thres_lower=None, b_thres_upper=None, min_signal=None, include_b0=False):
        
        import dipy.reconst.dki as dki
        import dipy.data

        #### gtab
        b_mask = (bvals > b_thres_lower) * (bvals < b_thres_upper)
        b_mask[bvals < 100] = include_b0
        print('b_mask: ', b_mask)
        gtab = dipy.data.gradient_table(bvals[b_mask], bvecs=bvecs[b_mask, :],
                                        big_delta=big_delta, small_delta=small_delta, 
                                        b0_threshold=b0_threshold)

        #### Instantiate the Tensor model
        dkimodel = dki.DiffusionKurtosisModel(gtab)

        #### Fit the data
        dkifit = dkimodel.fit(volume[:, :, :, b_mask])
        
        self.dkifit = dkifit

        self.dki_FA = dkifit.fa
        self.dki_MD = dkifit.md
        self.dki_AD = dkifit.ad
        self.dki_RD = dkifit.rd