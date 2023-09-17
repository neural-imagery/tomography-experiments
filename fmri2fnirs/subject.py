import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from geometry import Geometry
import utils
from skimage import measure
import json
import jdata as jd
import plotly.graph_objects as go


class Subject(object):
    """
    This class defines a NSD subject.
    """
    
    def __init__(self, id):
        self.id = id
        self.path = f"data/sub{self.id}/anat/"
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
        T1_file    = f"T1_{resolution}_masked"
        seg_ext    = "_head_seg.nii.gz"
        
        if os.path.isfile(self.path+T1_file+seg_ext): 
            seg_data = nib.load(self.path+T1_file+seg_ext).get_fdata()
        else:

            print("Creating anatomical segmentation...")
            # create directory
            if not os.path.exists(self.path): os.makedirs(self.path)
        
            # get anatomical NSD data
            url  = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/subj{self.id}/anat/"
            head_mask_data = utils.download_nsd_file(url, self.path, f"brainmask_{resolution}", return_numpy=True)
            brain_seg      = utils.get_brain_seg(url, self.path, T1_file)

            # create anatomical segmentation 
            seg_data = np.zeros(head_mask_data.shape)
            seg_data[head_mask_data==1] = 1
            seg_data[brain_seg.get_fdata()==1] = 2
            seg_data[brain_seg.get_fdata()==2] = 3
            seg_data[brain_seg.get_fdata()==3] = 4
            seg = nib.Nifti1Image(seg_data, brain_seg.affine, brain_seg.header)
            nib.save(seg, self.path+T1_file+seg_ext)

        return seg_data
    

    ###########################################################################
    # Optodes
    ###########################################################################

    def place_optodes(self, nsources: int = 10, ndetectors: int = 100, 
                        detrad : float = 3):
        """
        Get optode locations from brain segmentation
        """
        
        if hasattr(self, 'geometry'):
            if self.geometry.nsources == nsources and self.geometry.ndetectors == ndetectors:
                print("Optodes already placed")
                return
        
        # pad z layer with layer of zeros for better edges
        segmentation_padded = utils.padz_with0layer(self.segmentation)

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
    
    
    ###########################################################################
    # Visualization
    ###########################################################################

    def display_setup(self, seg=None, geom=None):
        
        if seg is None: seg = self.segmentation
        if geom is None: geom = self.geometry

        nx, ny, nz = seg.shape
        def get_lims_colors(surfacecolor):
            return np.min(surfacecolor), np.max(surfacecolor)
        def get_the_slice(x, y, z, surfacecolor):
            return go.Surface(x=x, y=y, z=z, surfacecolor=surfacecolor, coloraxis='coloraxis')
        def colorax(vmin, vmax):
            return dict(cmin=vmin, cmax=vmax)
        
        # plot z slice
        x = np.arange(nx); y = np.arange(ny); x, y = np.meshgrid(x,y)
        z_idx = nz//2; z = z_idx * np.ones(x.shape)
        surfcolor_z = seg[:, :, z_idx].T
        sminz, smaxz = get_lims_colors(surfcolor_z)
        slice_z = get_the_slice(x, y, z, surfcolor_z)
        
        # plot y slice
        x = np.arange(nx); z = np.arange(nz); x, z = np.meshgrid(x,z)
        y_idx = ny//3; y = y_idx * np.ones(x.shape)
        surfcolor_y = seg[:, y_idx, :].T
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
    
    
    ###########################################################################
    # Transform
    ###########################################################################

    def transform(self, sessionID, runID):
        """
        Bring the segmentation and optodes into the run's functional space.
        """

        resolution = "1pt0"
        func_path = f"data/sub{self.id}/func/fmri/sess{sessionID}/run{runID}/"
        T1_file   = f"T1_{resolution}_masked"
        fmri_file = f"sub-{self.id}_ses-nsd{sessionID}_task-nsdcore_run-{runID}_bold"
        seg_file = "head_seg"
        ext = ".nii.gz"

        if os.path.isfile(func_path+seg_file+".npy"): seg = np.load(func_path+seg_file+".npy")
        else: 

            # create directory
            if not os.path.exists(func_path): os.makedirs(func_path)    

            # get functional NSD data
            url = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata_rawdata/sub-{self.id}/ses-nsd{sessionID}/func/"
            utils.download_nsd_file(url, func_path, fmri_file)

            # register anatomical segmention in functional space using FSLs `flirt`
            # anat_seg_file = f"data/sub{self.id}/anat/T1_{resolution}_masked_head_seg.nii.gz"
            # os.system(f"flirt -in {anat_seg_file} -ref {func_path}{fmri_file}{ext} -out {func_path}{seg_file}{ext}")

            # get transform matrix
            os.system(f"flirt -in {self.path}{T1_file}{ext} -ref {func_path}{fmri_file}{ext} -omat {func_path}anat2func.mat")

            # apply transform to segmentation
            os.system(f"flirt -in {self.path}{T1_file}_{seg_file}{ext} -ref {func_path}{fmri_file}{ext} -applyxfm -init {func_path}anat2func.mat -out {func_path}{seg_file}{ext}")

            # load and save functional segmentation
            seg = np.round(nib.load(func_path+seg_file+ext).get_fdata()).astype('uint8')
            np.save(func_path+seg_file+".npy", seg)
        
        geometry = utils.transform_geometry(self, seg)
        utils.save_optodes_json(seg, geometry)
            
        return seg, geometry

        
        


