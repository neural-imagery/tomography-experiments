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