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

def region_to_mua(region, optical_properties):
    mua = np.zeros_like(region, dtype=np.float64)
    for i in range(len(optical_properties)):
        mua[region == i] = optical_properties[i,0]
    return mua

def detweight(detp, prop):
    """
    Calculates the detector weights for each measurement.

    Parameters
    ----------
    detp : numpy.ndarray
        The detector path information from pmcx.run().
    prop : numpy.ndarray
        The optical properties of each medium.

    Returns
    -------
    numpy.ndarray
        The detector weights for each measurement.
    """
    # For each measurement, we multiply the path length in each medium by the absorption coefficient, and exponentiate
    # to get the intensity.

    pathlengths = detp['ppath']
    absorption_coefficients = np.array(prop)[1:, 0] # exclude background

    weights = np.exp(-pathlengths @ absorption_coefficients) #* cfg['unitinmm']) # (nmeas,)

    return weights

def get_cw_data(res: dict, cfg: dict):
    """
    Converts the output of pmcx.mcxlab() into a (ndetectors,) np.ndarray

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
    ndetectors = cfg['detpos'].shape[0]
    weights = detweight(detp, cfg['prop'])

    intensities = np.bincount(detp['data'][0].astype('int64'), weights=weights, minlength=(ndetectors+1))[1:] # there is no detector 0

    return intensities


def dettime(detp, prop, unitinmm=1):
    """
    Recalculate the detected photon time using partial path data and
    optical properties (for perturbation Monte Carlo or detector readings).

    Parameters:
    detp (dict): The second output from mcxlab. detp must be a dictionary.
    prop (list): Optical property list, as defined in the cfg.prop field of mcxlab's input.
    unitinmm (float): Voxel edge-length in mm. If ignored, assume to be 1 (mm).

    Returns:
    dett (numpy.ndarray): Recalculated detected photon time based on the partial path data and optical property table.
    """

    R_C0 = 3.335640951981520e-12  # inverse of light speed in vacuum

    # Check the number of media
    medianum = len(prop)

    dett = np.zeros(detp['ppath'].shape[0])
    for i in range(medianum - 1):
        refractive_index = prop[i + 1][3]  # refractive index
        dett += refractive_index * detp['ppath'][:, i] * R_C0 * unitinmm
    return dett

def get_td(res, prop):
    """
    Calculate the time-domain data from the partial path data.

    Parameters:
    res (dict): The first output from mcxlab. res must be a dictionary.
    prop (list): Optical property list, as defined in the cfg.prop field of mcxlab's input.

    Returns:
    td (numpy.ndarray): Time-domain data.
    """

    # Check the number of media
    medianum = len(prop)

    # Get the number of detectors
    detnum = res['detp']['detid'].shape[0]

    # Initialize the time-domain data
    td = np.zeros((detnum, res['nphoton']))

    # Get the detector weights
    weights = detweight(res['detp'], prop)

    # Get the arrival times for each detector
    arrival_times_by_detector = get_arrival_times(res['detp'], prop)

    # Calculate the time-domain data
    for det_id, arrival_times in arrival_times_by_detector.items():
        td[det_id, :] = np.histogram(arrival_times, bins=res['nphoton'], weights=weights[det_id])[0]

    return td