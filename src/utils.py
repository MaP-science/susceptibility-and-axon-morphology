import numpy as np



def get_angles_between_vectors(vecs, vec):

    dot = np.sum(vecs*vec,axis=-1)

    radians = np.arccos(np.abs(dot / (np.linalg.norm(vecs, axis=-1) * np.linalg.norm(vec))))

    return np.rad2deg(radians)