import os
import nibabel as nib
import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from scipy.linalg import svd

from skimage import measure
from geometry import Geometry

import jdata as jd
import json

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pmcx


###############################################################################
# Segmentation
###############################################################################

def download_nsd_file(url, path, file, ext=".nii.gz", return_numpy=False):
    """
    Download file from NSD database if not already present.

    Parameters
    ----------
    url : str
    path : str
    file : str
    """
    if not os.path.isfile(path+file+ext):
        os.system(f"wget -P {path} {url}{file}{ext}")
    if return_numpy: return nib.load(path+file+ext).get_fdata()


def get_brain_seg(url, path, T1_file, ext=".nii.gz"):
    """
    Get brain segmentation from T1 image.

    Parameters
    ----------
    url : str
    path : str
    T1_file : str

    Returns
    -------
    brain_seg : nibabel.nifti1.Nifti1Image
    """
    
    # get T1 image
    download_nsd_file(url, path, T1_file)
    
    # segment brain from T1 image using FSLs `bet` (~5s)
    brain_ext = "_brain"
    if not os.path.isfile(path+T1_file+brain_ext+ext):
        os.system(f"bet {path}{T1_file}{ext} {path}{T1_file}{brain_ext} -m")

    # segment csf, grey and white matter from brain using FSLs `fast` (~5m8s)
    if not os.path.isfile(path+T1_file+"_seg"+ext):
        os.system(f"fast {path}{T1_file}{brain_ext}")

    brain_seg = nib.load(path+T1_file+brain_ext+"_seg"+ext)
    return brain_seg

###############################################################################
# Geometry
###############################################################################

# def get_probes_rand(n: int, vertices):
#     probes = vertices[
#         np.random.choice(np.arange(len(vertices)), 4 * n)
#     ]  # randomly over sample vertices
#     vertices_z = vertices[:, 2]
#     vertices_mid = (
#         min(vertices_z) + max(vertices_z)
#     ) / 2  # mid-point, only put electrodes on top half of brain
#     probes_z = probes[:, 2]
#     probes = probes[np.where(probes_z >= vertices_mid)]  # throw away bottom half
#     probes = probes[:n]
#     c = np.mean(vertices, axis=0)  # Coordinate of center of mass
#     probes = np.array(
#         [i * 0.99 + c * 0.01 for i in probes]
#     )  # bring sources a tiny bit into the brain 1%, make sure in tissue
#     return probes

def _euclidean_distance(a,b):
    return np.linalg.norm(a-b)

# uniformly sampled
def get_probes(n: int, vertices, frac_closer = 0.03):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(vertices)
    labels = kmeans.predict(vertices)
    centroids = kmeans.cluster_centers_
    central_points = []
    for i in range(n):
        cluster = vertices[labels == i]
        distances = [_euclidean_distance(c, centroids[i]) for c in cluster]
        central_points.append(cluster[np.argmin(distances)])
    com = vertices.mean(axis=0) # center of mass
    print(f"Probes placed... Bringing toward center in by {100*frac_closer}%")
    central_points = np.array(
        [i * (1-frac_closer) + com * frac_closer for i in central_points]
    )
    central_points = np.asarray(central_points)
    return central_points 


def get_normals(sources, vertices):
    c = np.mean(vertices, axis=0)  # center point
    normals = []
    kdtree = KDTree(vertices)
    for em in sources:
        _, neighbours_indices = kdtree.query(em, k=10)
        neighbours_10 = vertices[neighbours_indices]
        mean_point = np.mean(neighbours_10, axis=0)
        centered_points = neighbours_10 - mean_point
        _, _, Vt = svd(centered_points)
        normal_vector = Vt[-1, :]
        normal_vector /= np.linalg.norm(normal_vector)
        # we want our normals to point inwards
        if normal_vector @ (c - em) < 0:
            normal_vector *= -1
        normals.append(normal_vector)
    return np.asarray(normals)

def padz_with0layer(vol: np.ndarray):
    """
    Add a layer of zeros at the top so that the surface finds the top of the brain
    """
    return np.vstack([vol.T, np.zeros([1, vol.shape[1], vol.shape[2]])]).T


def transform_geometry(subj, seg_transformed):
    # Anatomy positions
    detpos_anat = subj.geometry.detectors[:, :3]
    shapea = subj.segmentation.shape
    vertex_anat, _, _, _ = measure.marching_cubes(
        np.vstack([subj.segmentation, np.zeros([1, shapea[1], shapea[2]])]), level=0.5)
    centroid_anat = np.mean(vertex_anat, axis=0)

    # Functional positions
    shapef = seg_transformed.shape
    vertex_functional, _, _, _ = measure.marching_cubes(
        np.vstack([seg_transformed, np.zeros([1, shapef[1], shapef[2]])]), level=0.5)
    centroid_func = np.mean(vertex_functional, axis=0)

    # Vector from anatomic to functional centroid
    vec_anat2func = centroid_func - centroid_anat

    # Volume calculations
    seg_anat = subj.segmentation.copy()
    seg_anat[seg_anat > 0] = 1
    vol_anat = np.sum(seg_anat.flatten())

    seg_func = seg_transformed.copy()
    seg_func[seg_func > 0] = 1
    vol_func = np.sum(seg_func.flatten())

    # Scale calculation
    scale = (vol_func / vol_anat)**(1 / 3)

    # Transform points
    def transform_a2b(points_a, center_a, center_b, scale):
        return (points_a - center_a) * scale + center_b

    detpos_func = transform_a2b(detpos_anat, centroid_anat, centroid_func, scale)
    detpos_func = np.hstack([detpos_func, np.ones([detpos_func.shape[0], 1])])

    srcpos_func = transform_a2b(subj.geometry.sources, centroid_anat, centroid_func, scale)
    srcdir_func = subj.geometry.directions

    geometry = Geometry(srcpos_func, detpos_func, srcdir_func)

    return geometry


###############################################################################
# MCX
###############################################################################

def get_detector_data(res: dict, cfg: dict):
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

    pathlengths = detp[1:]
    absorption_coefficients = np.array(cfg['prop'])[1:, 0] # exclude background

    intensities = np.exp(-absorption_coefficients @ pathlengths * cfg['unitinmm']) # (nmeas,)

    data = np.bincount(detp[0].astype('int64'), weights=intensities, minlength=(ndetectors+1))[1:] # there is no detector 0
    # data = np.bincount(detp[0].astype('int64'), minlength=ndetectors)

    return data


def run_baseline(seg, geom, props, src_idx, display_flux=False, time_idx=None):

    cfg = {
        'nphoton': 100000000,
        'vol': seg,
        'tstart': 0,
        'tend': 1e-8,
        'tstep': 1e-8,
        'srcpos': geom.sources[src_idx],
        'srcdir': geom.directions[src_idx],
        'prop': props,
        'detpos': geom.detectors,
        'replaydet':-1,
        'issavedet': 1,
        'issrcfrom0': 1,
        'issaveseed': 1,
        'unitinmm': 1.8,
        'maxdetphoton': 1000000000
        }

    res = pmcx.run(cfg)
    data = get_detector_data(res, cfg)
    
    if display_flux:
        if time_idx is None: time_idx = 0
        display_3d(res['flux'][...,time_idx], geom)

    return data, res, cfg


def compute_jacobian(res, cfg, geometry, display_jacobian=False, time_idx=None):
    
    cfg['seed']       = res['seeds']  # one must define cfg['seed'] using the returned seeds
    cfg['detphotons'] = res['detp']   # one must define cfg['detphotons'] using the returned detp data
    cfg['outputtype'] = 'jacobian'    # tell mcx to output absorption (μ_a) Jacobian

    if display_jacobian:
        if time_idx is None: time_idx = 0
        display_3d(res['flux'][:,:,:,time_idx], geometry)

    return pmcx.run(cfg)


###############################################################################
# fMRI BOLD
###############################################################################

def _boldpercent2optical(bold_change, seg_transformed):
    """
    Converts 1/BOLD percent change to optical properties change (µ_a)

    We use the rough estimates from Fig. 5 in:
    https://www.nmr.mgh.harvard.edu/optics/PDF/Strangman_NeuroImage_17_719_2002.pdf

    Parameters
    ----------
    bold_change : 4D numpy array
        4D array of shape (x, y, z, t)
    seg_transformed : 3D numpy array
        3D array of shape (x, y, z)
    media_properties : list of lists
        Each list contains [mu_a, mu_s, g, n]

    Returns
    -------
    dabs_690 : 3D numpy array of absorption changes at 690 nm
        3D array of shape (x, y, z)
    dabs_850 : 3D numpy array of absorption changes at 850 nm
        3D array of shape (x, y, z)

    """
    # only update in white & gray matter
    bold_change = bold_change * (seg_transformed[..., np.newaxis] > 2)

    # 1. estimate change in hb/hbO (10% change in 1/BOLD -> -2.5 µM change in hb, 1.25 µM change in hbO2)
    hb_change = bold_change * -2.5 / 0.1 * 1e-6 # Molar
    hbO2_change = bold_change * 1.25 / 0.1 * 1e-6 # Molar

    # 2. compute absorption coefficient change at 2 wavelengths (690 nm and 850 nm)
    # Source: Irving & Bigio, Quantitative Biomedical Optics
    dabs_690 = 492.2 * hb_change + 71.89 * hbO2_change # (mm^-1) at 690 nm
    dabs_850 = 181.0 * hb_change + 266.9 * hbO2_change # (mm^-1) at 850 nm

    return dabs_690, dabs_850


def fmri2optical(fmri, seg_transformed):
    """
    fmri: 4D numpy array
    anat_seg: 3D numpy array
    media_properties: list of lists, each list contains [mu_a, mu_s, g, n]
    """
    fmri_inv = 1 / (fmri + 1e-9)
    fmri_inv_avg = np.average(fmri_inv, axis=3)
    fmri_inv_percent = (fmri_inv - fmri_inv_avg[:,:,:,np.newaxis]) / fmri_inv_avg[:,:,:,np.newaxis]

    # clip to +/- 40%
    # fmri_inv_percent = np.clip(fmri_inv_percent, -0.4, 0.4)

    # plot_bold(seg_transformed, bold_percent[...,0])

    dabs_690, dabs_850 = _boldpercent2optical(fmri_inv_percent, seg_transformed)
    return dabs_690, dabs_850


def plot_bold(anat_seg, bold, slice=42):
    """
    anat_seg: segmented T1 image (nx, ny, nz) in a numpy array
    bold: bold percent change (nx, ny, nz) in a numpy array, e.g. 1 = no change
    slice: index of the slice along the z-axis
    """
    
    # Validate input dimensions
    if anat_seg.shape != bold.shape:
        raise ValueError("The shapes of anat_seg and bold must be the same")
    
    if slice < 0 or slice >= anat_seg.shape[2]:
        raise ValueError("Invalid slice index")
    
    # Extract the slice from the 3D images
    anat_slice = anat_seg[:,:,slice]
    bold_slice = bold[:,:,slice]

    # mask pixels where there's no brain
    bold_slice = np.ma.masked_where(anat_slice < 2, bold_slice)
    
    # Mask for pixels where there's a significant change in the BOLD signal
    mask = np.abs(bold_slice - 1) > 0.02

    # clip the BOLD percent change to +/- 20%
    bold_slice = np.clip(bold_slice, 0.8, 1.2)

    
    # Create the plot
    fig, ax = plt.subplots()
    
    # Show the anatomical slice
    ax.imshow(anat_slice, cmap='gray', origin='lower')
    
    # Overlay the BOLD percent change where it differs from 1
    im = ax.imshow(np.ma.masked_where(~mask, bold_slice-1), alpha=0.5, origin='lower', cmap='seismic')
    
    ax.set_title(f"BOLD Change Overlay on Slice {slice}")
    
    # Add a colorbar for the BOLD data
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('BOLD Percent Change')
    
    plt.show()

###############################################################################
# Visualization
###############################################################################

def display_3d(vol, geom):
    
    nx, ny, nz = vol.shape
    def get_lims_colors(surfacecolor):
        return np.min(surfacecolor), np.max(surfacecolor)
    def get_the_slice(x, y, z, surfacecolor):
        return go.Surface(x=x, y=y, z=z, surfacecolor=surfacecolor, coloraxis='coloraxis')
    def colorax(vmin, vmax):
        return dict(cmin=vmin, cmax=vmax)
    
    # plot z slice
    x = np.arange(nx); y = np.arange(ny); x, y = np.meshgrid(x,y)
    z_idx = nz//2; z = z_idx * np.ones(x.shape)
    surfcolor_z = vol[:, :, z_idx].T
    sminz, smaxz = get_lims_colors(surfcolor_z)
    slice_z = get_the_slice(x, y, z, surfcolor_z)
    
    # plot y slice
    x = np.arange(nx); z = np.arange(nz); x, z = np.meshgrid(x,z)
    y_idx = ny//3; y = y_idx * np.ones(x.shape)
    surfcolor_y = vol[:, y_idx, :].T
    sminy, smaxy = get_lims_colors(surfcolor_y)
    vmin = min([sminz, sminy])
    vmax = max([smaxz, smaxy])
    slice_y = get_the_slice(x, y, z, surfcolor_y)

    # plot points
    scatter_sources = go.Scatter3d(name='sources', x=geom.sources[:,0], 
                                    y=geom.sources[:,1], z=geom.sources[:,2], 
                                    mode='markers', marker=dict(size=3, color='red'))
    scatter_detectors = go.Scatter3d(name='detectors', x=geom.detectors[:,0], 
                                        y=geom.detectors[:,1], z=geom.detectors[:,2], 
                                        mode='markers', marker=dict(size=3, color='blue'))

    fig1 = go.Figure(data=[slice_z, slice_y, scatter_sources, scatter_detectors])
    fig1.update_layout(
            width=700, height=700,
            scene_xaxis_range=[0, nx], scene_yaxis_range=[0, ny], scene_zaxis_range=[0, nz], 
            coloraxis=dict(colorscale='deep', colorbar_thickness=25, colorbar_len=0.75,
                            **colorax(vmin, vmax)))
    fig1.show()


def animate_flux(res, seg_transformed):
    # Set up the figure
    zidx = seg_transformed.shape[2] // 2
    fig, ax = plt.subplots(figsize=(10, 10))

    # Initial frame
    im1 = ax.imshow(seg_transformed[:,:,zidx], animated=True, cmap='gray')
    im2 = ax.imshow(np.log10(res['flux'][:,:,zidx, 0]), animated=True, alpha=0.5, cmap='jet')

    # Update function
    def update(t):
        im2.set_array(np.log10(res['flux'][:,:,zidx, t]))
        return im2,

    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=res['flux'].shape[3], blit=True)

    # Save the animation
    ani.save('visualization.mp4', writer='ffmpeg', fps=5)

    plt.close(fig)