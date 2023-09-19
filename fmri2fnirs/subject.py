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
import pmcx


class Subject(object):
    """
    This class defines a NSD subject.
    """
    
    def __init__(self, id):
        self.id = id
        self.anat_path = f"data/sub{self.id}/anat/"
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
        
        if os.path.isfile(self.anat_path+T1_file+seg_ext): 
            seg_data = nib.load(self.anat_path+T1_file+seg_ext).get_fdata()
        else:

            print("Creating anatomical segmentation...")

            # create directory
            if not os.path.exists(self.anat_path): os.makedirs(self.anat_path)
        
            # get anatomical NSD data
            url  = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/subj{self.id}/anat/"
            head_mask_data = utils.download_nsd_file(url, self.anat_path, f"brainmask_{resolution}", return_numpy=True)
            brain_seg      = utils.get_brain_seg(url, self.anat_path, T1_file)

            # create anatomical segmentation 
            seg_data = np.zeros(head_mask_data.shape)
            seg_data[head_mask_data==1] = 1
            seg_data[brain_seg.get_fdata()==1] = 2
            seg_data[brain_seg.get_fdata()==2] = 3
            seg_data[brain_seg.get_fdata()==3] = 4
            seg = nib.Nifti1Image(seg_data, brain_seg.affine, brain_seg.header)
            nib.save(seg, self.anat_path+T1_file+seg_ext)

        return seg_data
    

    ###########################################################################
    # Optodes
    ###########################################################################

    def place_optodes(self, nsources: int = 10, ndetectors: int = 100, 
                        detrad : float = 3, frac_closer=0.03, display_setup=False):
        """
        Get optode locations from brain segmentation
        """
        
        self.geom_path = self.anat_path + f"fnirs_geometry_{nsources}_{ndetectors}_{detrad}_{frac_closer}/"
        if os.path.exists(self.geom_path): 
            sources = np.load(self.geom_path+'source_positions.npy')
            directions = np.load(self.geom_path+'source_directions.npy')
            detectors = np.load(self.geom_path+'detector_positions.npy')
            self.geometry = Geometry(sources, detectors, directions)

        else:
            
            print("Creating optode geometry...")

            # create directory
            os.makedirs(self.geom_path)
        
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
            
            # Find orthogonal directions of sources (pointing into brain)
            directions = utils.get_normals(sources, vertices)
            
            # create class attributes
            self.geometry = Geometry(sources, detectors, directions)

            # save source positions, source directions and detector positions 
            np.save(self.geom_path+'source_positions.npy', self.geometry.sources)
            np.save(self.geom_path+'source_directions.npy', self.geometry.directions)
            np.save(self.geom_path+'detector_positions.npy', self.geometry.detectors)

        if display_setup:
            utils.display_3d(self.segmentation, self.geometry)
    
    
    ###########################################################################
    # fMRI
    ###########################################################################
    
    def compute_jacobian(self, props, display_jacobian=False, time_idx=None):
        """
        Compute the Jacobian matrix for the subject's optode geometry.
        """
        
        for src_idx in range(1):#self.geometry.nsources):

            res, cfg = utils.forward(self.segmentation, self.geometry, props, src_idx=src_idx)
        
            cfg['seed']       = res['seeds']  # one must define cfg['seed'] using the returned seeds
            cfg['detphotons'] = res['detp']   # one must define cfg['detphotons'] using the returned detp data
            cfg['outputtype'] = 'jacobian'    # tell mcx to output absorption (μ_a) Jacobian

            res2 = pmcx.run(cfg)
            np.save(self.geom_path+f"jac_src{src_idx}.npy", np.squeeze(res2['flux']))

        if display_jacobian:
            if time_idx is None: time_idx = 0
            utils.display_3d(res2['flux'][:,:,:,time_idx], self.geometry)


    def transform(self, sessionID, runID, display_setup=False):
        """
        Bring the segmentation and optodes into the run's functional space.
        """

        resolution = "1pt0"
        func_path = f"data/sub{self.id}/func/sess{sessionID}/run{runID}/"
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
            os.system(f"flirt -in {self.anat_path}{T1_file}{ext} -ref {func_path}{fmri_file}{ext} -omat {func_path}anat2func.mat")

            # apply transform to segmentation
            # os.system(f"flirt -in {self.anat_path}{T1_file}_{seg_file}{ext} -ref {func_path}{fmri_file}{ext} -applyxfm -init {func_path}anat2func.mat -out {func_path}{seg_file}{ext}")
            # seg = np.round(nib.load(func_path+seg_file+ext).get_fdata()).astype('uint8')
            seg = utils.transform_volume(self.segmentation, np.loadtxt(func_path+"anat2func.mat"), output_shape=nib.load(func_path+fmri_file+ext).get_fdata().shape[:-1])
            seg = np.round(seg).astype('uint8')
            np.save(func_path+seg_file+".npy", seg)
        
        # transform geometry (with scaling 1/1.8 scaling)
        anat2func = np.loadtxt(func_path+"anat2func.mat")
        sources   = utils.transform_points(self.geometry.sources, anat2func)
        detectors = utils.transform_points(self.geometry.detectors[:,:3], anat2func)
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
        fmri = nib.load(f'data/sub{self.id}/func/sess{sessionID}/run{runID}/sub-{self.id}_ses-nsd{sessionID}_task-nsdcore_run-{runID}_bold.nii.gz').get_fdata()
        fmri_inv = 1 / (fmri + 1e-9)
        fmri_inv_avg = np.average(fmri_inv, axis=3)
        fmri_inv_percent = (fmri_inv - fmri_inv_avg[:,:,:,np.newaxis]) / fmri_inv_avg[:,:,:,np.newaxis]
        # clip to +/- 40%
        fmri_inv_percent = np.clip(fmri_inv_percent, -0.4, 0.4)
        # plot_bold(seg_transformed, bold_percent[...,0])
        dmua_690, dmua_850 = utils._boldpercent2optical(fmri_inv_percent, seg)
        
        return optical_baseline, dmua_850, dmua_690
    

    def get_jacobian(self, sessionID, runID, src_idx):
        """
        Get jacobian in functional space for a given source.
        """

        func_path = f"data/sub{self.id}/func/sess{sessionID}/run{runID}/"
        jac_file = f"jac_src{src_idx}"
        fmri_file = f"sub-{self.id}_ses-nsd{sessionID}_task-nsdcore_run-{runID}_bold"
        ext  = '.nii.gz'

        # get Jacobian computed in anatomical space
        J = np.load(self.geom_path+jac_file+'.npy')
        
        # transform Jacobian into functional space
        output_shape = nib.load(func_path+fmri_file+ext).get_fdata().shape[:-1]
        J_transformed = utils.transform_volume(J, np.loadtxt(func_path+"anat2func.mat"), output_shape=output_shape)
        J_transformed = np.reshape(J_transformed, (np.prod(output_shape), self.geometry.ndetectors))

        return J_transformed

