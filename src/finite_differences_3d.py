import torch



def dx_centered(array, mode='edge', step_size=2, res=1):
    """
    Computes the centered finite difference of the inputted array along the x
    direction. Ie. across rows.

    Parameters
    ----------
    array : np.ndarray
        The array of interest.

    Returns
    -------
    dx : np.ndarray
        The derivative of array.
    """

    if mode == 'roll':

        dx = (torch.roll(array, shifts=(-1), dims=(1)) - torch.roll(array, shifts=(1), dims=(1))) / step_size

    elif mode == 'no correction':

        dx = (array[:,step_size:,:] - array[:,:-step_size,:])/(step_size)

    elif mode == 'get units':

        dx = (torch.roll(array, shifts=(-1), dims=(1)) - torch.roll(array, shifts=(1), dims=(1))) / (res)

    else:
        dx = (array[:,step_size:,:] - array[:,:-step_size,:])/(step_size)


        #### padding
        # because "Only 3D, 4D, 5D padding with non-constant padding are supported"
        # for torch.nn.functional.pad() for now
        # TODO: Update when available.
        dx = _pad(dx, mode=mode, dim=1)

    return dx



def dy_centered(array, mode='edge', step_size=2, res=1):
    """
    Computes the centered finite difference of the inputted array along the y
    direction. Ie. across columns.

    Parameters
    ----------
    array : np.ndarray
        The array of interest.

    Returns
    -------
    dy : np.ndarray
        The derivative of array.
    """

    if mode == 'roll':

        dy = (torch.roll(array, shifts=(-1), dims=(0)) - torch.roll(array, shifts=(1), dims=(0))) / step_size

    elif mode == 'no correction':

        dy = (array[step_size:,:,:] - array[:-step_size,:,:])/(step_size)

    elif mode == 'get units':

        dy = (torch.roll(array, shifts=(-1), dims=(0)) - torch.roll(array, shifts=(1), dims=(0))) / (res)

    else:

        dy = (array[step_size:,:,:] - array[:-step_size,:,:])/(step_size)

        #### padding
        # because "Only 3D, 4D, 5D padding with non-constant padding are supported"
        # for torch.nn.functional.pad() for now
        # TODO: Update when available.
        dy = _pad(dy, mode=mode, dim=0)

    return dy



def dz_centered(array, mode='edge', step_size=2, res=1):
    """
    Computes the centered finite difference of the inputted array along the z
    direction. Ie. across depth.

    Parameters
    ----------
    array : np.ndarray
        The array of interest.

    Returns
    -------
    dz : np.ndarray
        The derivative of array.
    """

    if mode == 'roll':

        dz = (torch.roll(array, shifts=(-1), dims=(2)) - torch.roll(array, shifts=(1), dims=(2))) / step_size

    elif mode == 'no correction':

        dz = (array[:,:,step_size:] - array[:,:,:-step_size])/(step_size)

    elif mode == 'get units':

        dz = (torch.roll(array, shifts=(-1), dims=(2)) - torch.roll(array, shifts=(1), dims=(2))) / (res)

    else:

        dz = (array[:,:,step_size:] - array[:,:,:-step_size])/(step_size)

        #### padding
        # because "Only 3D, 4D, 5D padding with non-constant padding are supported"
        # for torch.nn.functional.pad() for now
        # TODO: Update when available.
        dz = _pad(dz, mode=mode, dim=2)

    return dz



def dxdx_centered(array, mode='edge'):

    dxdx = (array[:, 2:, :] - 2*array[:, 1:-1, :] + array[:, :-2, :])

    return dxdx



def dydy_centered(array, mode='edge'):

    dydy = (array[2:, :, :] - 2*array[1:-1, :, :] + array[:-2, :, :])

    return dydy



def dzdz_centered(array, mode='edge'):

    dzdz = (array[: , :, 2:] - 2*array[:, :, 1:-1] + array[:, :, :-2])

    return dzdz



def _pad(array, mode=None, dim=None):

    assert mode != None, 'mode was not defined.'
    assert dim != None, 'dim was not defined.'

    if mode == 'edge':
        if dim == 0:
            f = array[:1, :, :] #first row
            l = array[-1:, :, :] #last row
            array = torch.cat((f,array,l), dim=dim)
        elif dim == 1:
            f = array[:, :1, :] #first column
            l = array[:, -1:, :] #last column
            array = torch.cat((f,array,l), dim=dim)
        elif dim == 2:
            f = array[:, :, :1] #first column
            l = array[:, :, -1:] #last column
            array = torch.cat((f,array,l), dim=dim)
    elif mode == 'circular':
        if dim == 0:
            l = array[:1, :, :] #first row
            f = array[-1:, :, :] #last row
            array = torch.cat((f,array,l), dim=dim)
        elif dim == 1:
            l = array[:, :1, :] #first column
            f = array[:, -1:, :] #last column
            array = torch.cat((f,array,l), dim=dim)
        elif dim == 2:
            l = array[:, :, :1] #first column
            f = array[:, :, -1:] #last column
            array = torch.cat((f,array,l), dim=dim)
    elif mode == 'modified edge':
        if dim == 0:
            l = array[:1, :, :] #first row
            f = array[-1:, :, :] #last row
            array = torch.cat((f,array,l), dim=dim)
        elif dim == 1:
            l = array[:, :1, :] #first column
            f = array[:, -1:, :] #last column
            array = torch.cat((f,array,l), dim=dim)
        elif dim == 2:
            l = array[:, :, :1] #first column
            f = array[:, :, -1:] #last column
            array = torch.cat((f,array,l), dim=dim)
    else:
        print('mode is unknown.')

    return array
