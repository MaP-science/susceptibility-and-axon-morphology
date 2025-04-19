import numpy as np
import matplotlib.pyplot as plt
import ipywidgets

class ImageSliceViewer3D:
    """
    NB: Always use "%matplotlib notebook Otherwise it doesn't really work.

    Adapted from: https://github.com/mohakpatel/ImageSliceViewer3D

    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks.

    User can interactively change the slice plane selection for the image and
    the slice plane being viewed.

    Arguments:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html

    """

    def __init__(self, volumes, title=None, alphas=None, figsize=(4,4), cmap='Greys_r',
                 cmin=None, cmax=None):
        
        self.volumes = volumes
        self.title = title
        self.figsize = figsize
        self.cmap = cmap
        self.cmin = cmin
        self.cmax = cmax

        if alphas == None:
            self.alphas = [1.0,] * len(self.volumes)
        else:
            self.alphas = alphas

        # Call to select slice plane
        _ = ipywidgets.interact(
            self.view_selection, 
            view=ipywidgets.RadioButtons(
                options=['x-y','y-z', 'z-x'], 
                value='x-y',
                description='Slice plane:', 
                disabled=False,
            )
        )

    def view_selection(self, view):

        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z":[2,1,0], "z-x":[0,2,1], "x-y": [0,1,2]}

        self.vols = []

        for volume in self.volumes:
            try:
                self.vols.append(np.transpose(volume, orient[view]))
            except:
                # in case of RGB which has a fourth channel dimension
                self.vols.append(np.transpose(volume, orient[view] + [3]))

        maxZ = self.vols[0].shape[2] - 1

        # Call to view a slice within the selected slice plane
        ipywidgets.interact(self.plot_slice,
            z=ipywidgets.IntSlider(
                min=0, 
                max=maxZ, 
                step=1, 
                continuous_update=False,
                description='Slice index:',
                layout=ipywidgets.Layout(width='95%')
            ),
        )

    def plot_slice(self, z):

        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)

        plt.title(self.title)

        for vol, alpha in zip(self.vols, self.alphas):
            
            plt.imshow(
                vol[:,:,z], 
                cmap=plt.get_cmap(self.cmap), 
                alpha=alpha,
                vmin=self.cmin, 
                vmax=self.cmax
            )

        plt.show()
