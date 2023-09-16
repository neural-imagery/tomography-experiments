import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from geometry import Geometry
import utils
from skimage import measure
import json
import jdata as jd


class Subject(object):
    """
    This class defines a NSD subject.
    """
    
    def __init__(self, id, ):
        self.id = id
        self.segmentation = self._get_seg()
    
    ###########################################################################
    # Segmentation
    ###########################################################################

    def _get_seg(self):
        """
        Get a 1mm resolution head segmentation in anatomical space for the 
        specified NSD subject. The index-to-structure mapping is as follows:
        
        0: background (air)
        1: scalp and skull
        2: csf
        3: grey matter
        4: white matter

        Returns
        -------
        seg_data : 3D numpy array of shape (x, y, z)
        """
        resolution = "1pt0"
        path       = f"data/sub{self.id}/anat/"
        T1_file    = f"T1_{resolution}_masked"
        seg_ext    = "_head_seg.nii.gz"
        
        if os.path.isfile(path+T1_file+seg_ext): seg_data = nib.load(path+T1_file+seg_ext).get_fdata()
        else:

            print("Creating anatomical segmentation...")
            # create directory
            if not os.path.exists(path): os.makedirs(path)
        
            # get anatomical NSD data
            url  = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/subj{self.id}/anat/"
            head_mask_data = utils.download_nsd_file(url, path, f"brainmask_{resolution}", return_numpy=True)
            brain_seg      = utils.get_brain_seg(url, path, T1_file)

            # create anatomical segmentation 
            seg_data = np.zeros(head_mask_data.shape)
            seg_data[head_mask_data==1] = 1
            seg_data[brain_seg.get_fdata()==1] = 2
            seg_data[brain_seg.get_fdata()==2] = 3
            seg_data[brain_seg.get_fdata()==3] = 4
            seg = nib.Nifti1Image(seg_data, brain_seg.affine, brain_seg.header)
            nib.save(seg, path+T1_file+seg_ext)

        return seg_data
    
    def plot_segmentation(self, slice=None):
        """
        Plot a slice of the segmentation.
        """
        if not slice: slice = self.segmentation.shape[2]//2
        plt.figure(figsize=(6,6))
        plt.imshow(self.segmentation[:,:,slice]); plt.colorbar()
        plt.title('Anatomical segmentation')
        plt.show()
    

    ###########################################################################
    # Optodes
    ###########################################################################

    def get_optodes(self, nsources: int = 10, ndetectors: int = 100, 
                        detrad : float = 3):
        """
        Get optode locations from brain segmentation
        """
        
        if hasattr(self, 'geometry'):
            if self.geometry.nsources == nsources and self.geometry.ndetectors == ndetectors:
                print("Optodes already placed")
                return
        
        segmentation_padded = utils.padz_with0layer(self.segmentation)  # pad z layer with layer of zeros for better edges

        # make mesh
        vertices, _, _, _ = measure.marching_cubes(segmentation_padded, level=0.5)
        vertices = np.asarray(vertices, "float")
        # only select top-half of the vertices 
        c = np.mean(vertices,axis=0) # center
        vertices = vertices[(vertices[:,2] > c[2]) | ((vertices[:,2] - c[2]) > (vertices[:,1] - c[1]))]

        # place optodes uniformly
        detectors = utils.get_probes(ndetectors, vertices)  # Initiate 100 detectors

        # add radius to detectors
        detectors = np.hstack([detectors, np.ones([len(detectors), 1]) * detrad])
        sources = utils.get_probes(nsources, vertices)  # Initiate 10 probes
        directions = utils.get_normals(
            sources, vertices
        )  # Find orthogonal directions of sources (pointing into brain)
        
        self.geometry = Geometry(sources, detectors, directions)

    def plot_optodes(self):
        
        if not hasattr(self, 'geometry'):
            print("Place optodes")
            return
        fig = plt.figure(figsize=(6,6))
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.geometry.sources[:,0], self.geometry.sources[:,1], 
                     self.geometry.sources[:,2], c='r', label='sources')
        ax.scatter3D(self.geometry.detectors[:,0], self.geometry.detectors[:,1], 
                     self.geometry.detectors[:,2], c='b', label='detectors')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.legend(); plt.show()

    def save_optodes_json(
            self,
            vol_name="test.json",
            json_boilerplate: str = "colin27.json"):
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
            self.segmentation, {"compression": "zlib", "base64": 1}
        )  # serialize volume
        # manipulate binary str ing format so that it can be turned into json
        vol_encoded["_ArrayZipData_"] = str(vol_encoded["_ArrayZipData_"])[2:-1]

        with open(json_boilerplate) as f:
            json_inp = json.load(f)  # Load boilerplate json to dict
        json_inp["Shapes"] = vol_encoded  # Replaced volume ("shapes") atribute
        json_inp["Session"]["ID"] = vol_name  # and model ID
        # Make optode placement
        sources_list = []
        for s, d in zip(self.geometry.sources, self.geometry.directions):
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
        for d in self.geometry.detectors:
            detectors_list.append({"Pos": [d[0], d[1], d[2]], "R": d[3]})
        json_inp["Optode"] = {
            "Source": sources_list[
                0
            ],  # For the json we just pick one source, just for mcx viz
            "Detector": detectors_list,
        }
        json_inp["Domain"]["Dim"] = [
            int(i) for i in self.segmentation.shape
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
    
    
    ###########################################################################
    # Transform
    ###########################################################################

    def transform(self, sessionID, runID):
        """
        Bring the segmentation and optodes into the run's functional space.
        """

        path = f"data/sub{self.id}/func/fmri/sess{self.id}/run{runID}/"
        fmri_file = f"sub-{self.id}_ses-nsd{sessionID}_task-nsdcore_run-{runID}_bold"
        seg_ext = "_head_seg"
        ext = ".nii.gz"

        if os.path.isfile(path+fmri_file+seg_ext+".npy"): seg = np.load(path+fmri_file+seg_ext+".npy")
        else: 

            # create directory
            if not os.path.exists(path): os.makedirs(path)    

            # get head segmentation in anatomical space
            _, anat_seg_file = get_anat_segmentation(subjectID, resolution, plot=False)

            # get functional NSD data
            url = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata_rawdata/sub-{subjectID}/ses-nsd{sessionID}/func/"
            download_nsd_file(url, path, fmri_file)

            # TODO: Update anat_seg_file
            # register anatomical segmention in functional space using FSLs `flirt`
            # anat_seg_file = f"data/sub{subjectID}/anat/T1_{resolution}_masked_head_seg.nii.gz"
            os.system(f"flirt -in {anat_seg_file} -ref {path}{fmri_file}{ext} -out {path}{fmri_file}{seg_ext}{ext}")

            # load and save functional segmentation
            seg = np.round(nib.load(path+fmri_file+seg_ext+ext).get_fdata()).astype('uint8')
            np.save(path+fmri_file+seg_ext+".npy", seg)

        # plot segmentation   
        if plot:
            plt.imshow(seg[:,:,seg.shape[2]//2]); plt.colorbar()
            plt.title('Segmentation in functional space')
            
        return seg

        
        


