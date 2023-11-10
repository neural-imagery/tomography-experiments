import numpy as np
from pmcx import dettime

def add_ball_to_array(array, center_mm, radius_mm, val, grid_resolution_mm=1):
    """
    Adds a ball of specified value to a 3D numpy array, given a physical resolution of the grid.

    Parameters:
    -----------
    array : numpy.ndarray
        A 3D numpy array where the ball will be added.
    center_mm : tuple of floats
        The (x, y, z) coordinates of the ball's center in millimeters.
    radius_mm : float
        The radius of the ball in millimeters.
    val : float
        The value with which the ball's volume will be filled.
    grid_resolution_mm : float, optional
        The physical size (in millimeters) of each grid point in the array. Default is 1 mm.

    Returns:
    --------
    numpy.ndarray
        The 3D numpy array with the ball added.
    """
    
    # Convert center and radius from millimeters to grid points
    center = tuple(coord / grid_resolution_mm for coord in center_mm)
    radius = radius_mm / grid_resolution_mm

    # Create a meshgrid for the array dimensions
    x, y, z = np.ogrid[0:array.shape[0], 0:array.shape[1], 0:array.shape[2]]

    # Calculate the Euclidean distance from each point in the meshgrid to the center
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    # Set the value for points within the specified radius
    array[distances <= radius] = val

    return array

def get_arrival_times(detp, prop):
    # Calculate arrival times for all photons
    all_arrival_times = dettime(detp, prop=prop) # arrival times
    all_arrival_times = all_arrival_times.flatten()

    # Get unique detector IDs and their indices
    unique_det_ids, indices = np.unique(detp['detid'], return_inverse=True)

    # Initialize a dictionary to store arrival times by detector
    arrival_times_by_detector = {det_id: [] for det_id in unique_det_ids}

    # Vectorized segregation of arrival times by detector ID
    for det_id in unique_det_ids:
        arrival_times_by_detector[det_id] = all_arrival_times[indices == det_id]
    
    return arrival_times_by_detector

# based off of get_detector_data in fmri2fnirs/util.py
def get_cw_data(res: dict, cfg: dict):
    """
    Converts the output of pmcx.run() into a (ndetectors,) np.ndarray

    Parameters
    ----------
    res : dict
        output of pmcx.run()
    cfg : dict
        configuration dictionary used to run pmcx

    Returns
    -------
    data : (ndetectors,) np.ndarray
    """
    detp = res['detp']
    
    # detp[0] is the detector id, detp[1:nprops] is the path length in each medium

    # For each measurement, we multiply the path length in each medium by the absorption coefficient, and exponentiate
    # to get the intensity.

    ndetectors = cfg['detpos'].shape[0]

    pathlengths = detp['ppath']
    absorption_coefficients = np.array(cfg['prop'])[1:, 0] # exclude background

    weights = np.exp(-pathlengths @ absorption_coefficients) #* cfg['unitinmm']) # (nmeas,)

    # Get unique detector IDs and their indices
    unique_det_ids, indices = np.unique(detp['detid'], return_inverse=True)

    intensities = np.bincount(detp['data'][0].astype('int64'), weights=weights, minlength=(ndetectors+1))[1:] # there is no detector 0

    # Create a dictionary mapping detector IDs to intensities
    intensities_by_detector = {det_id: intensities[idx] for idx, det_id in enumerate(unique_det_ids)}
    
    return intensities_by_detector
