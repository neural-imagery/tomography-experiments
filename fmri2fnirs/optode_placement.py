# Code to loads 3d numpy array and boilerplate json and (a) can dump JSON file, agumented to feed to mcx cloud and (b) the cfg
# dictionary to feed to python bindings

import numpy as np
import json
import jdata as jd

from scipy import ndimage
from skimage import measure
from scipy.spatial import KDTree
from scipy.linalg import svd


def get_probes(n: int, vertices):
    probes = vertices[
        np.random.choice(np.arange(len(vertices)), 4 * n)
    ]  # randomly over sample vertices
    vertices_z = vertices[:, 2]
    vertices_mid = (
        min(vertices_z) + max(vertices_z)
    ) / 2  # mid-point, only put electrodes on top half of brain
    probes_z = probes[:, 2]
    probes = probes[np.where(probes_z >= vertices_mid)]  # throw away bottom half
    probes = probes[:n]
    c = np.mean(vertices, axis=0)  # Coordinate of center of mass
    probes = np.array(
        [i * 0.99 + c * 0.01 for i in probes]
    )  # bring sources a tiny bit into the brain 1%, make sure in tissue
    return probes


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
    return np.vstack([vol.T, np.zeros([1, 120, 120])]).T


def get_optodes(vol: np.ndarray, nsources: int = 10, ndetectors: int = 100, detrad : float = 3):
    """
    Get optode locations from brain segmentation

    Parameters
    ----------
    vol : np.ndarray
        3d numpy array specifying different layers of the brain
    detrad : float
        Radius of detectors

    Returns
    -------
    sources : list of np.ndarray
        List of vertices (x,y,z) where we initiated sources
    detectors : list of np.ndarray
        List of vertices (x,y,z) where we initiated detectors
    directions : list of np.ndarray
        List of unit vectors (x,y,z), pointint normal to brian towards center of brain
    """

    vol = padz_with0layer(vol)  # pad z layer with layer of zeros for better edges

    # make mesh
    vertices, faces, _, _ = measure.marching_cubes(vol, level=0.5)
    vertices = np.asarray(vertices, "float")

    # zthresh = (max(vertices[:,2]) + min(vertices[:,2]))/2
    # zmask = vertices[:,2] > zthresh
    # vert_half = vertices[zmask]
    # face_indices_to_keep = np.isin(faces, np.where(zmask)).all(axis=1)
    # filtered_faces = faces[face_indices_to_keep]

    # place optodes randomly
    detectors = get_probes(ndetectors, vertices)  # Initiate 100 detectors

    # add radius to detectors
    detectors = np.hstack([detectors, np.ones([len(detectors), 1]) * detrad])

    sources = get_probes(nsources, vertices)  # Initiate 10 probes
    directions = get_normals(
        sources, vertices
    )  # Find orthogonal directions of sources (pointing into brain)
    return sources, detectors, directions


def save_optodes_json(
    sources,
    detectors,
    directions,
    vol,
    vol_name="test.json",
    json_boilerplate: str = "colin27.json",
):
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
        vol, {"compression": "zlib", "base64": 1}
    )  # serialize volume
    # manipulate binary string format so that it can be turned into json
    vol_encoded["_ArrayZipData_"] = str(vol_encoded["_ArrayZipData_"])[2:-1]

    with open(json_boilerplate) as f:
        json_inp = json.load(f)  # Load boilerplate json to dict
    json_inp["Shapes"] = vol_encoded  # Replaced volume ("shapes") atribute
    json_inp["Session"]["ID"] = vol_name  # and model ID
    # Make optode placement
    # sources,detectors,directions = get_optodes(vol)
    sources_list = []
    for s, d in zip(sources, directions):
        sources_list.append(
            {
                "Type": "pencil",
                "Pos": [s[0], s[1], s[2]],
                "Dir": [d[0], d[1], d[2], 0],
                "Param1": [0, 0, 0, 0],  # cargo cult, dont know what this does
                "Param2": [0, 0, 0, 0],
            }
        )
    detectors_list = []
    for d in detectors:
        detectors_list.append({"Pos": [d[0], d[1], d[2]], "R": d[3]})
    json_inp["Optode"] = {
        "Source": sources_list[
            0
        ],  # For the json we just pick one source, just for mcx viz
        "Detector": detectors_list,
    }
    json_inp["Domain"]["Dim"] = [
        int(i) for i in vol.shape
    ]  # Set the spatial domain of Simulation
    with open(vol_name, "w") as f:
        json.dump(json_inp, f, indent=4)  # Write above changes to file

    ### Build Config File
    # Get the layer properties [mua,mus,g,n] from default colin27
    prop = [list(i.values()) for i in json_inp["Domain"]["Media"]]
    # position detectors in a way python bindings like
    detpos = [i["Pos"] + [i["R"]] for i in detectors_list]
    ### Build cfg dict
    cfg = {
        "nphoton": int(1e7),
        "vol": vol,
        "tstart": 0,
        "tend": 5e-9,
        "tstep": 5e-9,
        "srcpos": sources_list[0]["Pos"],
        "srcdir": sources_list[0]["Dir"],
        "prop": prop,
        "detpos": detpos,  # to detect photons, [x,y,z,radius]
        "issavedet": 1,  # not sure how important the rest below this line is
        "issrcfrom0": 1,  # flag ensure src/det coordinates align with voxel space
        "issaveseed": 1,  # set this flag to store dtected photon seed data
    }
    return  # cfg,json_inp,sources_list


# The cfg dic will only have one source, the first source from sources_list
