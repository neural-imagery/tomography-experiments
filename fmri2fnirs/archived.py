import numpy as np
from scipy.ndimage import convolve

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

def compute_jacobian(res, cfg, geometry, display_jacobian=False, time_idx=None):
    
    cfg['seed']       = res['seeds']  # one must define cfg['seed'] using the returned seeds
    cfg['detphotons'] = res['detp']   # one must define cfg['detphotons'] using the returned detp data
    cfg['outputtype'] = 'jacobian'    # tell mcx to output absorption (μ_a) Jacobian

    if display_jacobian:
        if time_idx is None: time_idx = 0
        display_3d(res['flux'][:,:,:,time_idx], geometry)

    return pmcx.run(cfg)

def transform_volume(vol, matrix, output_shape, scaling=1/1.8):
    
    # create scaling matrix
    scaling_matrix = np.array([[scaling, 0, 0, 0],
                               [0, scaling, 0, 0],
                               [0, 0, scaling, 0],
                               [0, 0, 0, 1]])
    
    # combine scaling matrix with the transform matrix
    combined_transform_matrix = np.dot(scaling_matrix, matrix)
    
    # create a coordinate grid for the output shape
    x, y, z = np.meshgrid(np.arange(output_shape[0]),
                          np.arange(output_shape[1]),
                          np.arange(output_shape[2]), indexing='ij')
    
    # apply inverse transform to find corresponding coordinates in input_array
    homogeneous_coordinates = np.stack((x, y, z, np.ones(output_shape)), axis=-1)
    inverse_transform = np.linalg.inv(combined_transform_matrix)
    input_coordinates = np.einsum('...ij,...j->...i', inverse_transform, homogeneous_coordinates)
    
    # separate x, y, z coordinates
    x_in, y_in, z_in, _ = np.split(input_coordinates, 4, axis=-1)

    # clip coordinates to be within the valid range
    x_in = np.clip(x_in, 0, vol.shape[0] - 1)
    y_in = np.clip(y_in, 0, vol.shape[1] - 1)
    z_in = np.clip(z_in, 0, vol.shape[2] - 1)
    
    # perform trilinear interpolation
    x0 = np.floor(x_in).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y_in).astype(int)
    y1 = y0 + 1
    z0 = np.floor(z_in).astype(int)
    z1 = z0 + 1
    
    # clip to be within array bounds
    x1 = np.clip(x1, 0, vol.shape[0] - 1)
    y1 = np.clip(y1, 0, vol.shape[1] - 1)
    z1 = np.clip(z1, 0, vol.shape[2] - 1)
    
    # trilinear interpolation
    wa = (x1 - x_in) * (y1 - y_in) * (z1 - z_in)
    wb = (x1 - x_in) * (y1 - y_in) * (z_in - z0)
    wc = (x1 - x_in) * (y_in - y0) * (z1 - z_in)
    wd = (x1 - x_in) * (y_in - y0) * (z_in - z0)
    we = (x_in - x0) * (y1 - y_in) * (z1 - z_in)
    wf = (x_in - x0) * (y1 - y_in) * (z_in - z0)
    wg = (x_in - x0) * (y_in - y0) * (z1 - z_in)
    wh = (x_in - x0) * (y_in - y0) * (z_in - z0)

    vol_out = np.squeeze(wa * vol[x0, y0, z0] + wb * vol[x0, y0, z1] + \
              wc * vol[x0, y1, z0] + wd * vol[x0, y1, z1] + \
              we * vol[x1, y0, z0] + wf * vol[x1, y0, z1] + \
              wg * vol[x1, y1, z0] + wh * vol[x1, y1, z1])
    
    return vol_out

import numpy as np