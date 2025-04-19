import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider



def get_angles_between_vectors(vecs, vec):

    dot = np.sum(vecs*vec,axis=-1)

    radians = np.arccos(np.abs(dot / (np.linalg.norm(vecs, axis=-1) * np.linalg.norm(vec))))

    return np.rad2deg(radians)



def plot_3d_slices(data):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Get dimensions
    depth, height, width = data.shape
    
    # Create initial plot
    slice_idx = depth // 2
    im = ax.imshow(data[slice_idx], cmap='viridis')
    plt.colorbar(im)
    
    # Add slider
    ax_slice = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slice, 'Slice', 0, depth-1, valinit=slice_idx,
                   valstep=1, valfmt='%d')
    
    def update(val):
        idx = int(slider.val)
        ax.clear()
        ax.imshow(data[idx], cmap='viridis')
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()