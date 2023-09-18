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

import matplotlib.pyplot as plt
from scipy.ndimage import convolve

import matplotlib.animation as animation


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

def save_optodes_json(segmentation, geometry, vol_name="test.json", json_boilerplate: str = "colin27.json"):
    """
    Serializes and saves json input to mcx for isual display on mcx cloud. ++other stuff

    Parameters
    ----------
    sources : list of np.ndarray

    numpy_fname : str
        Path to .npy file containing 3d numpy array
    json_boilerplate : str
        Path to json boilerplate file.
    Returns
    -------
    cfg : dict
        Config dictionary to feed to python
    json_inp : dict
        JSON dictionary, input to mcx simulations
    sources_list : list
        List of dictionaries containing sources information (postion and direction)
    """
    # # Load numpy file
    # vol = np.load(numpy_fname)
    # vol_name = numpy_fname[:-4] # name of volume file

    ### JSON manipulation
    # encode & compress vol
    vol_encoded = jd.encode(
        np.asarray(segmentation + 0.5, dtype=np.uint8), {"compression": "zlib", "base64": 1}
    )  # serialize volume
    # manipulate binary str ing format so that it can be turned into json
    vol_encoded["_ArrayZipData_"] = str(vol_encoded["_ArrayZipData_"])[2:-1]

    with open(json_boilerplate) as f:
        json_inp = json.load(f)  # Load boilerplate json to dict
    json_inp["Shapes"] = vol_encoded  # Replaced volume ("shapes") atribute
    json_inp["Session"]["ID"] = vol_name  # and model ID
    # Make optode placement
    sources_list = []
    for s, d in zip(geometry.sources, geometry.directions):
        sources_list.append(
            {
                "Type": "pencil",
                "Pos": [s[0], s[1], s[2]],
                "Dir": [d[0], d[1], d[2], 0],
                "Param1": [0, 0, 0, 0],  # cargo cult, dont know what param1 and 2 does
                "Param2": [0, 0, 0, 0],
            }
        )
    detectors_list = []
    for d in geometry.detectors:
        detectors_list.append({"Pos": [d[0], d[1], d[2]], "R": d[3]})
    json_inp["Optode"] = {
        "Source": sources_list[
            0
        ],  # For the json we just pick one source, just for mcx viz
        "Detector": detectors_list,
    }
    json_inp["Domain"]["Dim"] = [
        int(i) for i in segmentation.shape
    ]  # Set the spatial domain of Simulation
    with open(vol_name, "w") as f:
        json.dump(json_inp, f, indent=4)  # Write above changes to file
    print(f"Saved to {vol_name}")

    # ### Build Config File
    # # Get the layer properties [mua,mus,g,n] from default colin27
    # prop = [list(i.values()) for i in json_inp["Domain"]["Media"]]
    # # position detectors in a way python bindings like
    # detpos = [i["Pos"] + [i["R"]] for i in detectors_list]
    # 
    #### Build cfg dict
    # cfg = {
    #     "nphoton": int(1e7),
    #     "vol": self.segmentation,
    #     "tstart": 0,
    #     "tend": 5e-9,
    #     "tstep": 5e-9,
    #     "srcpos": sources_list[0]["Pos"],
    #     "srcdir": sources_list[0]["Dir"],
    #     "prop": prop,
    #     "detpos": detpos,  # to detect photons, [x,y,z,radius]
    #     "issavedet": 1,  # not sure how important the rest below this line is
    #     "issrcfrom0": 1,  # flag ensure src/det coordinates align with voxel space
    #     "issaveseed": 1,  # set this flag to store dtected photon seed data
    # }
    return  # cfg,json_inp,sources_list

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
# MCX
###############################################################################

def get_detector_data_from_flux(flux, points_3d, detector_radius=5):
    """
    flux: 4D numpy array
    points_3d: 2D numpy array of shape (n, 3)
    detector_size: int (mm)
    """
    kernel_size = int(detector_radius / 1.8)
    kernel = np.ones((kernel_size, kernel_size, kernel_size))
    flux_conv = np.zeros_like(flux)

    # Convolve each time slice with the 3D kernel
    for t in range(flux.shape[-1]):
        flux_conv[:, :, :, t] = convolve(flux[:, :, :, t], kernel, mode='constant', cval=0.0)
        
    # Extract the time components at the specified 3D points
    x_indices = points_3d[:, 0].astype(int)
    y_indices = points_3d[:, 1].astype(int)
    z_indices = points_3d[:, 2].astype(int)

    return flux_conv[x_indices, y_indices, z_indices, :]

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


# fMRI

def get_optical_baseline(seg_transformed, media_properties):
    # optical_baseline should be of shape (2, x,y,z)
    newshape = (2, *seg_transformed.shape)
    optical_baseline = np.zeros(newshape)
    for idx, prop in enumerate(media_properties):
        # when seg_transformed == idx, optical_baseline[0] == prop[0], optical_baseline[1] == prop[1]
        optical_baseline[0][seg_transformed == idx] = prop[0] # mu_a
        optical_baseline[1][seg_transformed == idx] = prop[1] # mu_s
    return optical_baseline

def _boldpercent2optical(bold_change, seg_transformed, media_properties):
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
    bold_change = np.ma.masked_where(seg_transformed <= 2, bold_change)

    # 1. estimate change in hb/hbO (10% change in 1/BOLD -> -2.5 µM change in hb, 1.25 µM change in hbO2)
    hb_change = bold_change * -2.5 / 0.1 * 1e-6 # Molar
    hbO2_change = bold_change * 1.25 / 0.1 * 1e-6 # Molar

    # 2. compute absorption coefficient change at 2 wavelengths (690 nm and 850 nm)
    # Source: Irving & Bigio, Quantitative Biomedical Optics
    dabs_690 = 4922 * hb_change + 718.9 * hbO2_change # 690 nm
    dabs_850 = 1810 * hb_change + 2669 * hbO2_change # 850 nm

    return dabs_690, dabs_850


def fmri2optical(fmri, seg_transformed, media_properties):
    """
    fmri: 4D numpy array
    anat_seg: 3D numpy array
    media_properties: list of lists, each list contains [mu_a, mu_s, g, n]
    """
    fmri_avg = np.average(fmri, axis=3)
    bold_percent = (fmri - fmri_avg[:,:,:,np.newaxis]) / fmri_avg[:,:,:,np.newaxis]
    one_over_bold_percent = 1 / bold_percent

    # plot_bold(seg_transformed, bold_percent[...,0])

    dabs_690, dabs_850 = _boldpercent2optical(one_over_bold_percent, seg_transformed, media_properties)
    return dabs_690, dabs_850

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

    data = np.bincount(detp[0].astype('int64'), weights=intensities, minlength=ndetectors)


    # data = np.bincount(detp[0].astype('int64'), minlength=ndetectors)
    return data