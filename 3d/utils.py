import numpy as np

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