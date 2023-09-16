import os
import nibabel as nib
import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from scipy.linalg import svd

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