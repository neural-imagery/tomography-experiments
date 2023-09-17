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

from scipy.ndimage import convolve


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
def get_probes(n: int, vertices):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(vertices)
    labels = kmeans.predict(vertices)
    centroids = kmeans.cluster_centers_
    central_points = []
    for i in range(n):
        cluster = vertices[labels == i]
        distances = [_euclidean_distance(c, centroids[i]) for c in cluster]
        central_points.append(cluster[np.argmin(distances)])
    com = vertices.mean(axis=0) # center of mass
    print("Probes placed... Bringing toward center in by 3%")
    central_points = np.array(
        [i * 0.97 + com * 0.03 for i in central_points]
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


###############################################################################
# MCX
###############################################################################

def get_detector_data(flux, points_3d, kernel_size=5):
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