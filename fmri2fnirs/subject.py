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
import time


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
        return np.array([str(i).zfill(2) for i in range(7, nsessions+1)])

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
        Place optodes uniformly on head given the subject's brain segmentation.

        Parameters
        ----------
        nsources : int
            Number of sources
        ndetectors : int
            Number of detectors
        detrad : float
            Detector radius (in mm)
        frac_closer : float
            Percent for which to bring optodes closer to the center
        display_setup : bool
            Whether to display the optode setup
        """
        
        self.geom_path = self.anat_path + f"fnirs_geometry_{nsources}_{ndetectors}_{detrad}_{frac_closer}/"
        if os.path.exists(self.geom_path): 
            sources = np.load(self.geom_path+'source_positions.npy')
            directions = np.load(self.geom_path+'source_directions.npy')
            detectors = np.load(self.geom_path+'parent_detector_positions.npy')
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
            
            # create geometry
            self.geometry = Geometry(sources, detectors, directions)

            # save source positions, source directions and detector positions 
            np.save(self.geom_path+'source_positions.npy', self.geometry.sources)
            np.save(self.geom_path+'source_directions.npy', self.geometry.directions)
            np.save(self.geom_path+'parent_detector_positions.npy', self.geometry.detectors)

            # initialize source and closest detectors subset
            self.geometry.initialize_src_dets(self.geom_path)

        if display_setup: utils.display_3d(self.segmentation, self.geometry)
    
    
    ###########################################################################
    # MCX
    ###########################################################################
    
    def compute_jacobian(self, props):
        """
        Compute the absorption (μ_a) Jacobian (nsrc, nx, ndet) for the subject's 
        optode geometry.

        Parameters
        ----------
        props : list of tuples
            Optical properties (μ_a, μ_s, g, n) for each tissue type in the 
            segmentation. 
        """

        # define paths and create directories
        data_path = self.geom_path + "baseline/" 
        jac_path  = self.geom_path + "jacobians/"
        if not os.path.exists(data_path): os.makedirs(data_path)
        if not os.path.exists(jac_path):  os.makedirs(jac_path)

        for src_idx in range(self.geometry.nsources):
            if not os.path.isfile(jac_path+f"jac_src{src_idx}.npy"):

                # run forward model and save detector data
                res, cfg = utils.forward(self.segmentation, self.geometry, props, src_idx=src_idx)
                data = utils.get_detector_data(res, cfg)
                np.save(data_path+f"base_src{src_idx}.npy", data)
            
                # define cfg['seed'] using the returned seeds
                cfg['seed'] = res['seeds']

                # define cfg['detphotons'] using the returned detp data
                cfg['detphotons'] = res['detp']

                # ensure output is absorption (μ_a) Jacobian
                cfg['outputtype'] = 'jacobian'

                res2 = pmcx.run(cfg)
                np.save(self.geom_path+f"jacobians/jac_src{src_idx}.npy", np.squeeze(res2['flux']))
    
    
    ###########################################################################
    # fMRI
    ###########################################################################

    def transform(self, sessionID, runID, jax=True, display_setup=False):
        """
        Transform the segmentation and optodes into the run's functional space.

        Parameters
        ----------
        sessionID : int
        runID : int
        display_setup : bool
            Whether to display the transformed segmentation and optode geometry.

        Returns
        -------
        seg_transformed : np.ndarray
            Transformed segmentation
        geometry : Geometry
            Transformed optode geometry
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

            # get anatomical-to-functional transform matrix
            os.system(f"flirt -in {self.anat_path}{T1_file}{ext} -ref {func_path}{fmri_file}{ext} -omat {func_path}anat2func.mat")
            anat2func = np.loadtxt(func_path+"anat2func.mat")
            output_shape = (120,120,84) # HARD-CODED to save time
            # output_shape = nib.load(func_path+fmri_file+ext).get_fdata().shape[:-1]
            os.remove(func_path+fmri_file+ext)

            # apply transform to segmentation
            if jax: seg = utils.transform_volume_jax(self.segmentation, anat2func, output_shape)
            else: seg = utils.transform_volume(self.segmentation, anat2func, output_shape)
            seg = np.round(seg).astype('uint8')
            np.save(func_path+seg_file+".npy", seg)
        
        # transform optode geometry
        scaling = 1.8
        anat2func = np.loadtxt(func_path+"anat2func.mat")
        sources   = utils.transform_points(self.geometry.sources, anat2func)
        detectors = utils.transform_points(self.geometry.detectors[:,:3], anat2func)
        detectors = np.hstack([detectors, scaling*self.geometry.detectors[:,3:]])
        directions = (anat2func[:3,:3]@self.geometry.directions.T).T
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        geometry = Geometry(sources, detectors, directions)
        # utils.save_optodes_json(seg, geometry)

        if display_setup: utils.display_3d(seg, geometry)
            
        return seg, geometry
    

    def get_optics(self, sessionID, runID, seg):
        """
        Get time-varying μ_a changes from optical baseline over the fMRI run.

        Parameters
        ----------
        sessionID : int
        runID : int
        seg : np.ndarray
            Segmentation
        props : list of tuples
            Optical properties (μ_a, μ_s, g, n) for each tissue type

        Returns
        -------
        dmua_850 : np.ndarray
            Time-varying μ_a changes at 850nm
        dmua_690 : np.ndarray
            Time-varying μ_a changes at 690nm
        """

        # changes in baseline over fMRI run
        fmri = nib.load(f'data/sub{self.id}/func/sess{sessionID}/run{runID}/sub-{self.id}_ses-nsd{sessionID}_task-nsdcore_run-{runID}_bold.nii.gz').get_fdata()
        fmri_inv = 1 / (fmri + 1e-9)
        fmri_inv_avg = np.average(fmri_inv, axis=3)
        fmri_inv_percent = (fmri_inv - fmri_inv_avg[:,:,:,np.newaxis]) / fmri_inv_avg[:,:,:,np.newaxis]
        
        # clip to +/- 40%
        fmri_inv_percent = np.clip(fmri_inv_percent, -0.4, 0.4)
        # plot_bold(seg_transformed, bold_percent[...,0])
        dmua_690, dmua_850 = utils._boldpercent2optical(fmri_inv_percent, seg)
        
        return dmua_850, dmua_690
    

    def get_jacobian(self, sessionID, runID, src_idx, jax=True):
        """
        Get μ_a Jacobian for a single source and transform into the run's functional 
        space.

        Parameters
        ----------
        sessionID : int
        runID : int
        src_idx : int

        Returns
        -------
        J_transformed : np.ndarray
        """

        # get Jacobian in anatomical space
        J = np.load(self.geom_path+f"jacobians/jac_src{src_idx}"+'.npy')
        
        # transform Jacobian into functional space
        func_path = f"data/sub{self.id}/func/sess{sessionID}/run{runID}/"
        anat2func = np.loadtxt(func_path+"anat2func.mat")
        output_shape = (120,120,84) # HARD-CODED to save time
        #output_shape = np.load(func_path+'head_seg.npy').shape
        ndetectors   = np.load(self.geom_path+f"detector_positions/detectors_src{src_idx}.npy").shape[0]
        
        if jax: 
            J_transformed = utils.transform_volume_jax(J, anat2func, output_shape=output_shape)
        else: 
            J_transformed = utils.transform_volume(J, anat2func, output_shape=output_shape)
        J_transformed = np.reshape(J_transformed, (np.prod(output_shape), ndetectors))

        return J_transformed

