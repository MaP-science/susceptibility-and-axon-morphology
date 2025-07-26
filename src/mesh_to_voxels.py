import torch
import pytorch3d as torch3d
from pytorch3d import io
import math
from tqdm.auto import tqdm
import numpy as np


##########################################################################################
##### DEMONSTRATED IN MORE DETAIL AT: https://github.com/sibowi/mesh_utils ###############
##########################################################################################



def check_ray_triangle_intersection(ray_origins, ray_direction, triangle, epsilon=1e-6):
    """
    Optimized to work for:
        >1 ray_origins
        1 ray_direction multiplied to match the dimension of ray_origins
        1 triangle

    Based on: Answer by BrunoLevy at
    https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
    Thank you!

    Parameters
    ----------
    ray_origin : torch.Tensor, (n_rays, n_dimensions), (x, 3)
    ray_directions : torch.Tensor, (n_rays, n_dimensions), (1, x)
    triangle : torch.Tensor, (n_points, n_dimensions), (3, 3)

    Return
    ------
    intersection : boolean (n_rays,)

    Test
    ----
    triangle = torch.Tensor([[0., 0., 0.],
                             [1., 0., 0.],
                             [0., 1., 0.],
                            ]).to(device)

    ray_origins = torch.Tensor([[0.5, 0.25, 0.25],
                                [5.0, 0.25, 0.25],
                               ]).to(device)

    ray_origins = torch.rand((10000, 3)).to(device)

    ray_direction = torch.Tensor([[0., 0., -10.],]).to(device)
    #ray_direction = torch.Tensor([[0., 0., 10.],]).to(device)

    ray_direction = ray_directions.repeat(ray_origins.shape[0], 1)

    check_ray_triangle_intersection(ray_origins, ray_direction, triangle)
    """

    E1 = triangle[1] - triangle[0] #vector of edge 1 on triangle
    E2 = triangle[2] - triangle[0] #vector of edge 2 on triangle
    N = torch.cross(E1, E2) # normal to E1 and E2

    invdet = 1. / -torch.einsum('ji, i -> j', ray_direction, N) # inverse determinant

    A0 = ray_origins - triangle[0]
    
    DA0 = torch.cross(A0, ray_direction)

    u = torch.einsum('ji, i -> j', DA0, E2) * invdet
    v = -torch.einsum('ji, i -> j', DA0, E1) * invdet
    t = torch.einsum('ji, i -> j', A0, N) * invdet

    intersection = (t >= 0.0) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)

    return intersection



def generate_voxel_grid(bbox, res):
    """
    Generates mesh grids where the points correspond to the corners of voxels.

    Parameters
    ----------
    bbox : tuple/list
        Content: xmin, xmax, ymin, ymax, zmin, zmax.
    res : float
        Resolution of the grid. Voxel-side-length.

    Return
    ------
    grid_x : torch.Tensor
    grid_y : torch.Tensor
    grid_z : torch.Tensor

    """

    if len(res) == 3:
        res_actual_x, res_actual_y, res_actual_z = res
    else:
        res_actual_x = res_actual_y = res_actual_z = res[0]

    xmin, xmax, ymin, ymax, zmin, zmax = bbox

    delta = 0.000001 # for numerical stuff

    xs = torch.arange(xmin, xmax-res_actual_x, res_actual_x)
    ys = torch.arange(ymin, ymax-res_actual_y, res_actual_y)
    zs = torch.arange(zmin, zmax-res_actual_z, res_actual_z)

    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs)

    return grid_x, grid_y, grid_z



def get_mesh_bbox(vertices):

    (xmin, ymin, zmin), _ = torch.min(vertices, dim=0)
    (xmax, ymax, zmax), _ = torch.max(vertices, dim=0)

    bbox = xmin, xmax, ymin, ymax, zmin, zmax

    return bbox



def get_ray_origins(vertices, res=0.1, bbox=None):

    device = vertices.device

    # generate grid
    grid_x, grid_y, grid_z = generate_voxel_grid(bbox, res)

    shape_grid = grid_x.shape

    # flatten for optimized computation
    grid_x_flat = grid_x.reshape(grid_x.numel(), 1)
    grid_y_flat = grid_y.reshape(grid_y.numel(), 1)
    grid_z_flat = grid_z.reshape(grid_z.numel(), 1)

    ray_origins = torch.cat((grid_x_flat, grid_y_flat, grid_z_flat), dim=1).to(device)

    return ray_origins, shape_grid


def get_ray_origins_masked(vertices, res=0.1, bbox=None, mask_prior=None):

    device = vertices.device

    # get bbox of mesh
    if bbox == None:
        bbox = get_mesh_bbox(vertices)

    # generate grid
    grid_x, grid_y, grid_z = generate_voxel_grid(bbox, res)

    shape_grid = grid_x.shape

    if mask_prior == None:
        mask_prior = torch.ones(shape_grid).type(torch.bool)

    grid_x, grid_y, grid_z = grid_x[mask_prior], grid_y[mask_prior], grid_z[mask_prior]

    # flatten for optimized computation
    grid_x_flat = grid_x.reshape(grid_x.numel(), 1)
    grid_y_flat = grid_y.reshape(grid_y.numel(), 1)
    grid_z_flat = grid_z.reshape(grid_z.numel(), 1)

    ray_origins = torch.cat((grid_x_flat, grid_y_flat, grid_z_flat), dim=1).to(device)

    return ray_origins, shape_grid



def dep_get_ray_direction(ray_direction, shape, device):

    ray_direction = torch.Tensor([ray_direction,]).to(device)

    ray_direction = ray_direction.repeat(shape, 1)

    return ray_direction



def get_ray_directions(ray_directions_singles, n_ray_origins, device):
    """
    Parameters
    ----------
    ray_directions_singles : list
    """

    ray_direction = ray_directions_singles.repeat_interleave(n_ray_origins, dim=0)

    return ray_direction


def get_faces_associated_with_given_ray_origins(faces, vertices, ray_origins, buffer_z_lower=0.0, buffer_z_higher=0.0, epsilon=0.0):
    """

    """

    #### no need to check faces that are not within the range of the ray_origins
    # doing batches along z
    # For a triangle to be associated with the slab of interest, one of the two conditions must be
    # fulfilled.
    #     1. >=1 of the vertice's z-coordinate must be >=slab_lower_z AND <= slab_upper_z
    # OR
    #     2. >=1 of the vertice's z-coordinate must be >=B_z AND >=1 point must be <=A_z

    (ray_origins_xmin, ray_origins_ymin, ray_origins_zmin), _ = torch.min(ray_origins, dim=0)
    (ray_origins_xmax, ray_origins_ymax, ray_origins_zmax), _ = torch.max(ray_origins, dim=0)

    # condition 1, if any vertex is contained by the slab
    mask_contained = (vertices[:, 2] >= (ray_origins_zmin - abs(buffer_z_lower) - epsilon)) * (vertices[:, 2] <= (ray_origins_zmax + buffer_z_higher + epsilon))

    mask_contained = torch.sum(mask_contained[faces], dim=1) > 0

    # condition 2, if one face is associated with vertices both above and below the slab
    condition_a = (vertices[:, 2] <= (ray_origins_zmin))
    condition_b = (vertices[:, 2] >= (ray_origins_zmax))

    mask_a = torch.sum(condition_a[faces], dim=1) > 0
    mask_b = torch.sum(condition_b[faces], dim=1) > 0

    mask_exceeding = mask_a * mask_b

    # final mask
    mask_faces_inside_range_oi = mask_contained + mask_exceeding

    faces_oi = faces[mask_faces_inside_range_oi]

    return faces_oi


def get_segmentation_from_mask_voxel_corners_inside(mask_inside):
    """
    Only if all 8 corners of a voxel are inside the mesh, the voxel is
    considered as being inside the mesh.

    """

    segmentation = mask_inside * \
                   torch.roll(mask_inside, shifts=(1), dims=(0)) * \
                   torch.roll(mask_inside, shifts=(1), dims=(1)) * \
                   torch.roll(mask_inside, shifts=(1, 1), dims=(0, 1)) * \
                   torch.roll(mask_inside, shifts=(1), dims=(2)) * \
                   torch.roll(mask_inside, shifts=(1, 1), dims=(2, 0)) * \
                   torch.roll(mask_inside, shifts=(1, 1), dims=(2, 1)) * \
                   torch.roll(mask_inside, shifts=(1, 1, 1), dims=(2, 0, 1))

    segmentation = torch.roll(segmentation, shifts=(-1, -1, -1), dims=(0, 1, 2))

    return segmentation



def batch(iterable, n=1):
    """
    Returns batch of size n.

    Parameters
    ----------
    iterable : iterable
        What ever you want batched.
    n : int
        Batch size.

    """

    l = len(iterable)

    for i in range(0, l, n):
        yield iterable[i:min(i + n, l)]



def get_angles_between_ray_directions_and_z(ray_directions):
    """
    Returns angles between ray_directions and z in radians.

    Parameters
    ----------
    ray_directions : torch.Tensor [n_ray_directions, n_dimensions]
       Ray directions.

    Returns
    -------
    theta_zs : torch.Tensor [n_ray_directions]
        Angle with respect to z for each vector in ray_directions.

    """

    def get_angles(a, b):

        inner_product = torch.einsum('ij, kj -> i', a, b) #(a * b)#.sum(dim=1)
        a_norm = a.pow(2).sum(dim=1).pow(0.5)
        b_norm = b.pow(2).sum(dim=1).pow(0.5)
        cos = inner_product / (a_norm * b_norm)
        angles = torch.acos(cos)

        angles = angles * 180. / np.pi

        return angles

    vector_z = torch.Tensor([0., 0., 1.]).unsqueeze(0).to(ray_directions.get_device())

    theta_zs = get_angles(ray_directions.type(torch.float32), vector_z.type(torch.float32))
    theta_zs = torch.deg2rad(theta_zs)

    return theta_zs



def get_buffer_lengths_z(theta_zs, shape_grid, res):
    """
    When ray_directions are not perpendicular with z, they might intersect with
    the mesh outside of the slab defined by the ray_origins. Hence, a buffer
    must be added to the area of the interval along z for which the faces are
    being checked.

    Parameters
    ----------
    theta_zs : torch.Tensor
        Angle with respect to z for each vector in ray_directions.
    shape_grid : tuple
        Shape of the grid of points that is being checked for being inside the
        mesh.
    res : float
        Spatial resolution of the grid of points.

    Returns
    -------
    buffer_lengths_z : torch.Tensor
        The z-projection for each theta_z over the maximal distance achievable
        (the diagonal) over the grid of points.

    """

    theta_xys = math.pi/2. - theta_zs

    diagonal = math.sqrt(shape_grid[0]**2 + shape_grid[1]**2) * res

    buffer_lengths_z = torch.tan(theta_xys) * diagonal

    return buffer_lengths_z



def get_mask_prior_from_segmentation_initial(segmentation_initial, n_neighbors, device=None):
    """
    When segmenting a mesh that has been obtained by cubify'ing a voxel-substrate
    and then decimating it, we don't have to perform segmentation of every point
    in a new voxel-grid. Instead we can perform segmentation only on selected
    points/voxels based on a mask_prior. This can save a lot of time!

    This function takes the segmentation_initial and n_neighbors as input, and
    returns a mask of voxels within a distance n_neighbors from the edges of
    segmentation_initial.

    Parameters
    ----------
    segmentation_initial : torch.Tensor
        Initial segmentation.
    n_neighbors : int
        Distance from edges in segmentation_initial to include in mask_prior.

    Returns
    -------
    mask_prior : torch.Tensor
        Mask of voxels within a distance n_neighbors from the edges of
        segmentation_initial.

    """

    # make dimensions fit
    segmentation_extended = torch.cat([segmentation_initial[-1:, :, :],segmentation_initial], dim=0)
    segmentation_extended = torch.cat([segmentation_extended[:, -1:, :], segmentation_extended], dim=1)
    segmentation_extended = torch.cat([segmentation_extended[:, :, -1:], segmentation_extended], dim=2)

    count = torch.zeros(segmentation_extended.shape).to(device)

    # count number of neighbors that are different from self
    for n_xdim in range(-n_neighbors, n_neighbors):
        for n_ydim in range(-n_neighbors, n_neighbors):
            for n_zdim in range(-n_neighbors, n_neighbors):
                count += ~torch.eq(segmentation_extended, torch.roll(segmentation_extended, shifts=(n_xdim, n_ydim, n_zdim), dims=(0, 1, 2)))

    mask_prior = count > 0. # get boolean mask from counts

    return mask_prior



def load_mesh(path_mesh):
    """
    Move somewhere else!
    """

    # get number of verts and faces
    with open(path_mesh, 'r') as file:

        lines = file.readlines()

        n_verts = int(lines[3].split(' ')[-1])
        n_faces = int(lines[7].split(' ')[-1])

    # load verts
    verts = np.loadtxt(path_mesh, skiprows=10, max_rows=n_verts)

    # load faces
    faces = np.loadtxt(path_mesh, skiprows=10+n_verts, max_rows=n_faces, usecols=(1, 2, 3)).astype(int)

    return torch.Tensor(verts), torch.Tensor(faces).to(torch.long)



def check_if_voxel_corners_are_inside_mesh(path_mesh, res, device, inspection_mode=False, bbox=None, verbose=False):
    """
    Main function. Combines
    """

    if len(res) == 3:
        res_actual_x, res_actual_y, res_actual_z = res
    else:
        res_actual_x = res_actual_y = res_actual_z = res[0]


    #### load mesh
    try:
        vertices, faces = torch3d.io.load_ply(path_mesh)
    except:
        vertices, faces = load_mesh(path_mesh)

    vertices = vertices.to(device).type(torch.float64)
    faces = faces.to(device)

    # get bbox of mesh
    if bbox == None:
        bbox = get_mesh_bbox(vertices)

    #### get tensor containing a point for each corner of each voxel
    ray_origins_all, shape_grid = get_ray_origins(vertices, res=res, bbox=bbox)
    ray_origins_all = ray_origins_all.type(torch.float64) ####
    print(f'Specs of points to be segmented: ')
    print(f'\t bbox = {bbox}')
    print(f'\t Shape of grid = {shape_grid}.')
    print(f'\t Points in total = {shape_grid.numel()}.')

    #### sort ray origins with respect their z-position
    mask_ray_origins_all_z_sorted = torch.argsort(ray_origins_all[:, 2])

    #### estimate suitable slab height
    # currently based on the most occuring z-length of all faces.
    # alternatives could be mean, median, max, ...

    # get the maximum distance between vertices in each face
    z_lengths_faces = torch.max(vertices[faces][:, :, -1], dim=1)[0] - torch.min(vertices[faces][:, :, -1], dim=1)[0]
    z_length_most_occuring = torch.mode(z_lengths_faces)[0]

    n_z_slices_per_batch = max(int((z_length_most_occuring // res_actual_z)), 1) # must be >=1

    # number of points per batch
    n = shape_grid[0] * shape_grid[1] * n_z_slices_per_batch

    #### for progress bar (tqdm)
    total = int(np.ceil(len(mask_ray_origins_all_z_sorted) / n))

    #### initialize
    ray_directions_singles = torch.Tensor([[10., 9., 0.], [-10., 9., res_actual_z/2], [10., -9., -res_actual_z/2]]).to(device).type(torch.float64)
    intersections = torch.zeros((ray_origins_all.shape[0], ray_directions_singles.shape[0])).to(device)

    #### get slab-buffer specs
    theta_zs = get_angles_between_ray_directions_and_z(ray_directions_singles)
    buffer_lengths_z = get_buffer_lengths_z(theta_zs, shape_grid, res_actual_z)

    #### voting policy
    # TODO: OPTIMIZE!
    # minimum number of rays per point to be classified as being inside the mesh
    # in order for that point to be considered as inside the mesh
    votes_min = np.ceil(len(ray_directions_singles)/2)

    #### do
    for idxs_oi in tqdm(batch(mask_ray_origins_all_z_sorted, n=n), total=total):

        ray_origins = ray_origins_all[idxs_oi, :]

        # get faces of interest
        faces_oi = get_faces_associated_with_given_ray_origins(faces, vertices,
                                                               ray_origins,
                                                               buffer_z_lower=min(buffer_lengths_z),
                                                               buffer_z_higher=max(buffer_lengths_z),
                                                               epsilon=0.0,
                                                              )

        if len(faces_oi) == 0:
            continue

        # get ray origins within an xy-boundary box of the mesh.
        xmin_vertices = torch.min(vertices[faces_oi][:, :, 0])
        ymin_vertices = torch.min(vertices[faces_oi][:, :, 1])
        xmax_vertices = torch.max(vertices[faces_oi][:, :, 0])
        ymax_vertices = torch.max(vertices[faces_oi][:, :, 1])

        mask_mesh_bbox = (ray_origins[:, 0] >= xmin_vertices) * (ray_origins[:, 0] <= xmax_vertices) * \
                         (ray_origins[:, 1] >= ymin_vertices) * (ray_origins[:, 1] <= ymax_vertices)

        # how many ray_origins are ignored
        if verbose: print(f'\t Checking {len(faces_oi)}/{len(faces)} faces. Ignoring {np.round(1 - torch.sum(mask_mesh_bbox).cpu().numpy()/len(mask_mesh_bbox), 2)} ray_origins in this slab.')

        ray_origins = ray_origins[mask_mesh_bbox, :]

        # repeat for linalg stuff
        ray_origins = ray_origins.repeat(len(ray_directions_singles), 1)

        # get ray direction
        # TODO: might want to use two rays that are perpendicular to each other. both in the plane.
        ray_directions = get_ray_directions(ray_directions_singles,
                                            n_ray_origins=ray_origins.shape[0]//len(ray_directions_singles),
                                            device=device)

        # go through all faces_oi
        for face in faces_oi:

            # get the corresponding vertices
            triangle = vertices[face].double()#.T

            # get votes of intersections
            votes = check_ray_triangle_intersection(ray_origins, ray_directions, triangle).double()

            votes = votes.reshape(len(ray_directions_singles), ray_origins.shape[0]//len(ray_directions_singles)).T #stupid work around becaused reshape ordering cannot be defined (as in np.reshape).

            intersections[idxs_oi[mask_mesh_bbox], :] += votes

    del votes, ray_origins, ray_directions
    torch.cuda.empty_cache()

    #### get boolean mask of whether points are inside or outside mesh
    mask_voxel_corners_inside = torch.sum((intersections % 2 == 1), dim=1) >= votes_min

    #### reshape
    mask_voxel_corners_inside = mask_voxel_corners_inside.reshape(shape_grid)

    #### return
    if inspection_mode:

        #reshape
        intersections = intersections.reshape(shape_grid + (len(ray_directions_singles),))
        intersections = [intersections[:, :, :, i] for i in range(intersections.shape[-1])]

        del ray_directions_singles
        torch.cuda.empty_cache()

        return mask_voxel_corners_inside, intersections
    else:
        ray_directions_singles
        torch.cuda.empty_cache()

        return mask_voxel_corners_inside


def check_if_voxel_corners_are_inside_mesh_with_mask(path_mesh, res, device, mask_prior=None, inspection_mode=False, bbox=None):
    """
    Main function. Combines
    """

    #### load mesh
    vertices, faces = torch3d.io.load_ply(path_mesh)
    vertices = vertices.to(device).type(torch.float64)
    faces = faces.to(device)

    #### get tensor containing a point for each corner of each voxel
    ray_origins_all, shape_grid = get_ray_origins_masked(vertices, res=res, bbox=bbox, mask_prior=mask_prior)
    ray_origins_all = ray_origins_all.type(torch.float64)

    if mask_prior == None:
        mask_prior = torch.ones(shape_grid).type(torch.bool)

    #### sort ray origins with respect their z-position
    mask_ray_origins_all_z_sorted = torch.argsort(ray_origins_all[:, 2])

    #### estimate suitable slab height
    # currently based on the most occuring z-length of all faces.
    # alternatives could be mean, median, max, ...
    z_lengths_faces = torch.max(vertices[faces][:, :, -1], dim=1)[0] - torch.min(vertices[faces][:, :, -1], dim=1)[0]
    z_length_most_occuring = torch.mode(z_lengths_faces)[0]

    n_z_slices_per_batch = max(int((z_length_most_occuring // res) / 1), 1) # must be >=1

    # number of points per batch
    if mask_prior == None:
        n = shape_grid[0] * shape_grid[1] * n_z_slices_per_batch
    else:
        n = torch.sum(mask_prior[:, :, 0]).to('cpu')

    #### for progress bar (tqdm)
    total = int(np.ceil(len(mask_ray_origins_all_z_sorted) / n))

    #### initialize
    ray_directions_singles = torch.Tensor([[10., 9., 1.], [-10., 9., 2.], [10., -9., -1.], [-10., -9., 1.], [3., -7., -1.]]).to(device).type(torch.float64)
    intersections = torch.zeros((ray_origins_all.shape[0], ray_directions_singles.shape[0])).to(device)

    #### get slab-buffer specs
    theta_zs = get_angles_between_ray_directions_and_z(ray_directions_singles)
    buffer_lengths_z = get_buffer_lengths_z(theta_zs, shape_grid, res) * 1e-3

    #### voting policy
    # TODO: OPTIMIZE!
    # minimum number of rays per point to be classified as being inside the mesh
    # in order for that point to be considered as inside the mesh
    votes_min = np.ceil(len(ray_directions_singles)/2)

    #### do
    for idxs_oi in tqdm(batch(mask_ray_origins_all_z_sorted, n=n), total=total):

        ray_origins = ray_origins_all[idxs_oi, :]

        # get faces of interest
        faces_oi = get_faces_associated_with_given_ray_origins(faces, vertices,
                                                               ray_origins,
                                                               buffer_z_lower=min(buffer_lengths_z),
                                                               buffer_z_higher=max(buffer_lengths_z),
                                                               epsilon=0.0, ####
                                                              )

        print(f'\t Checking {len(faces_oi)}/{len(faces)} faces.')

        if len(faces_oi) == 0:
            continue

        # get ray origins within an xy-boundary box of the mesh.
        xmin_vertices = torch.min(vertices[faces_oi][:, :, 0])
        ymin_vertices = torch.min(vertices[faces_oi][:, :, 1])
        xmax_vertices = torch.max(vertices[faces_oi][:, :, 0])
        ymax_vertices = torch.max(vertices[faces_oi][:, :, 1])

        mask_mesh_bbox = (ray_origins[:, 0] >= xmin_vertices) * (ray_origins[:, 0] <= xmax_vertices) * \
                         (ray_origins[:, 1] >= ymin_vertices) * (ray_origins[:, 1] <= ymax_vertices)

        # how many ray_origins are ignored
        print(f'\t Ignoring {np.round(1 - torch.sum(mask_mesh_bbox).cpu().numpy()/len(mask_mesh_bbox), 2)} ray_origins in this slab.')

        ray_origins = ray_origins[mask_mesh_bbox, :]

        # repeat for linalg stuff
        ray_origins = ray_origins.repeat(len(ray_directions_singles), 1)

        # get ray direction
        # TODO: might want to use two rays that are perpendicular to each other. both in the plane.
        ray_directions = get_ray_directions(ray_directions_singles,
                                            n_ray_origins=ray_origins.shape[0]//len(ray_directions_singles),
                                            device=device)

        # go through all faces_oi
        for face in faces_oi:

            # get the corresponding vertices
            triangle = vertices[face].double()#.T

            # get votes of intersections
            votes = check_ray_triangle_intersection(ray_origins, ray_directions, triangle).double()

            votes = votes.reshape(len(ray_directions_singles), ray_origins.shape[0]//len(ray_directions_singles)).T #stupid work around becaused reshape ordering cannot be defined (as in np.reshape).

            intersections[idxs_oi[mask_mesh_bbox], :] += votes

    del votes
    torch.cuda.empty_cache()

    #### get boolean mask of whether points are inside or outside mesh
    mask_voxel_corners_inside = torch.sum((intersections % 2 == 1), dim=1) >= votes_min

    #### reshape
    # convert from mask-space to substrate-space
    mask_voxel_corners_blank = torch.zeros(shape_grid).type(torch.bool).to(device)
    mask_voxel_corners_blank[mask_prior] = mask_voxel_corners_inside
    mask_voxel_corners_inside = mask_voxel_corners_blank
    # reshape
    mask_voxel_corners_inside = mask_voxel_corners_inside.reshape(shape_grid)

    #### return
    if inspection_mode:

        intersections_blank = torch.zeros(shape_grid + (len(ray_directions_singles),)).to(device)

        # convert from mask-space to substrate-space
        for i in range(intersections_blank.shape[-1]):
            intersections_blank[..., i][mask_prior] = intersections[:, i]

        #reshape
        intersections = intersections_blank.reshape(shape_grid + (len(ray_directions_singles),))
        intersections = [intersections[:, :, :, i] for i in range(intersections.shape[-1])]

        return mask_voxel_corners_inside, intersections
    else:
        return mask_voxel_corners_inside



def get_segmentation_from_mesh(path_ply, bbox, res, device, mode='None'):

    mask_voxel_corners = check_if_voxel_corners_are_inside_mesh(path_ply,
                                                                res=res,
                                                                device=device,
                                                                inspection_mode=False,
                                                                bbox=bbox)

    if mode == 'strictly inside':
        pass
    elif mode == 'strictly outside':
        mask_voxel_corners = torch.logical_not(mask_voxel_corners)
    elif mode == 'None':
        pass
    else:
        print('\n\n\nmode was not recognized...\n\n\n')

    # convert from point-space to voxel-space
    segmentation = get_segmentation_from_mask_voxel_corners_inside(mask_voxel_corners)

    segmentation = segmentation.to('cpu')

    #del segmentation
    del mask_voxel_corners

    #### more cleaning
    torch.cuda.empty_cache()

    return segmentation


def trim_segmentation(segmentation):

    """
    Does not preserve the geometry very well. But for 1-2 iterations on something
    with good resolution it is very fine.
    """

    segmentation_trimed = segmentation.clone()

    label_trim = 1
    label_extra = 0
    label_intra = 2

    # nearest neighbors (6)
    n0 = torch.roll(segmentation, 1, 0)
    n1 = torch.roll(segmentation, 1, 1)
    n2 = torch.roll(segmentation, 1, 2)
    n3 = torch.roll(segmentation, -1, 0)
    n4 = torch.roll(segmentation, -1, 1)
    n5 = torch.roll(segmentation, -1, 2)

    # # corners (8)
    # n6 = torch.roll(segmentation, shifts=(1, 1, 1), dims=(0, 1, 2))
    # n7 = torch.roll(segmentation, shifts=(-1, 1, 1), dims=(0, 1, 2))
    # n8 = torch.roll(segmentation, shifts=(-1, -1, 1), dims=(0, 1, 2))
    # n9 = torch.roll(segmentation, shifts=(1, -1, 1), dims=(0, 1, 2))
    # n10 = torch.roll(segmentation, shifts=(1, -1, -1), dims=(0, 1, 2))
    # n11 = torch.roll(segmentation, shifts=(1, 1, -1), dims=(0, 1, 2))
    # n12 = torch.roll(segmentation, shifts=(-1, -1, -1), dims=(0, 1, 2))
    # n13 = torch.roll(segmentation, shifts=(-1, 1, -1), dims=(0, 1, 2))
    #
    # # the sides between the corners (12)
    # n14 = torch.roll(segmentation, shifts=(1, 1), dims=(0, 1))
    # n15 = torch.roll(segmentation, shifts=(1, -1), dims=(0, 1))
    # n16 = torch.roll(segmentation, shifts=(-1, 1), dims=(0, 1))
    # n17 = torch.roll(segmentation, shifts=(-1, -1), dims=(0, 1))
    #
    # n18 = torch.roll(segmentation, shifts=(1, 1), dims=(0, 2))
    # n19 = torch.roll(segmentation, shifts=(1, -1), dims=(0, 2))
    # n20 = torch.roll(segmentation, shifts=(-1, 1), dims=(0, 2))
    # n21 = torch.roll(segmentation, shifts=(-1, -1), dims=(0, 2))
    #
    # n22 = torch.roll(segmentation, shifts=(1, 1), dims=(1, 2))
    # n23 = torch.roll(segmentation, shifts=(1, -1), dims=(1, 2))
    # n24 = torch.roll(segmentation, shifts=(-1, 1), dims=(1, 2))
    # n25 = torch.roll(segmentation, shifts=(-1, -1), dims=(1, 2))

    ns = [n0, n1, n2, n3, n4, n5, \
          # n6, n7, n8, n9, n10, n11, n12, n13, \
          # n14, n15, n16, n17, n18, n19, n20, n21, n22, n23, n24, n25,
         ]

    mask_trim = torch.sum(torch.stack([segmentation != n for n in ns]), dim=0) > 0
    mask_trim = mask_trim * (segmentation == label_trim)

    # if value True in mask, correct the value to the one that's occuring the most among the neighbors, and which is NOT label_trim
    count_neighbors_extra = torch.sum(torch.stack(ns) == label_extra, dim=0)
    count_neighbors_intra = torch.sum(torch.stack(ns) == label_intra, dim=0)

    segmentation_trimed[mask_trim * (count_neighbors_extra >= count_neighbors_intra)] = label_extra
    segmentation_trimed[mask_trim * (count_neighbors_extra < count_neighbors_intra)] = label_intra

    return segmentation_trimed
