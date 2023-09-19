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
from scipy.ndimage import affine_transform


class Subject(object):
    """
    This class defines a NSD subject.
    """
    
    def __init__(self, id):
        self.id = id
        self.path = f"data/sub{self.id}/anat/"
        self.sessions = self._init_sessions()
        self.runs = np.array([str(i).zfill(2) for i in range(1, 13)])
        self.segmentation = self._get_seg()
    
    def _init_sessions(self):
        if self.id in ['01','02','05','07']: nsessions = 40
        elif self.id in ['03', '06']: nsessions = 32
        elif self.id in ['04', '08']: nsessions = 30
        else: raise ValueError("Invalid subject ID")
        return np.array([str(i).zfill(2) for i in range(1, nsessions+1)])

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
                        detrad : float = 3, frac_closer=0.03, display_setup=False):
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
        detectors = utils.get_probes(ndetectors, vertices, frac_closer=frac_closer)

        # add radius to detectors
        detectors = np.hstack([detectors, np.ones([len(detectors), 1]) * detrad])
        sources = utils.get_probes(nsources, vertices, frac_closer=frac_closer)
        directions = utils.get_normals(sources, vertices)  # Find orthogonal directions of sources (pointing into brain)
        
        self.geometry = Geometry(sources, detectors, directions)

        if display_setup:
            utils.display_3d(self.segmentation, self.geometry)
    
    
    ###########################################################################
    # fMRI
    ###########################################################################
    
    def transform(self, sessionID, runID, display_setup=False):
        """
        Bring the segmentation and optodes into the run's functional space.
        """

        resolution = "1pt0"
        func_path = f"data/sub{self.id}/func/fmri/sess{sessionID}/run{runID}/"
        T1_file   = f"T1_{resolution}_masked"
        fmri_file = f"sub-{self.id}_ses-nsd{sessionID}_task-nsdcore_run-{runID}_bold"
        seg_file = "head_seg"
        ext = ".nii.gz"

        # transform segmentation
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
        

        # transform geometry with scaling from 1mm to 1.8mm
        def transform_points(M,v, scaling=1/1.8):
            v = np.hstack([v, np.ones((v.shape[0], 1))])
            return (M@v.T).T[:,:3] * scaling
        
        anat2func = np.loadtxt(func_path+"anat2func.mat")
        sources   = transform_points(anat2func, self.geometry.sources)
        detectors = transform_points(anat2func, self.geometry.detectors[:,:3])
        detectors = np.hstack([detectors, 2*self.geometry.detectors[:,3:]])
        directions = (anat2func[:3,:3]@self.geometry.directions.T).T
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        
        geometry = Geometry(sources, detectors, directions)
        # utils.save_optodes_json(seg, geometry)

        if display_setup: utils.display_3d(seg, geometry)
            
        return seg, geometry
    

    def get_optics(self, sessionID, runID, seg, props):
        """
        Get optical baseline and time-varying changes in optical baseline 
        (only mu_a for now) over the fMRI run.
        """
        
        # baseline
        newshape = (2, *seg.shape)
        optical_baseline = np.zeros(newshape)
        for idx, prop in enumerate(props):
            # when seg_transformed == idx, optical_baseline[0] == prop[0], optical_baseline[1] == prop[1]
            optical_baseline[0][seg == idx] = prop[0] # mu_a
            optical_baseline[1][seg == idx] = prop[1] # mu_s

        # changes in baseline over fMRI run
        fmri = nib.load(f'data/sub{self.id}/func/fmri/sess{sessionID}/run{runID}/sub-{self.id}_ses-nsd{sessionID}_task-nsdcore_run-{runID}_bold.nii.gz').get_fdata()
        fmri_inv = 1 / (fmri + 1e-9)
        fmri_inv_avg = np.average(fmri_inv, axis=3)
        fmri_inv_percent = (fmri_inv - fmri_inv_avg[:,:,:,np.newaxis]) / fmri_inv_avg[:,:,:,np.newaxis]
        # clip to +/- 40%
        fmri_inv_percent = np.clip(fmri_inv_percent, -0.4, 0.4)
        # plot_bold(seg_transformed, bold_percent[...,0])
        dmua_690, dmua_850 = utils._boldpercent2optical(fmri_inv_percent, seg)
        
        return optical_baseline, dmua_850, dmua_690

