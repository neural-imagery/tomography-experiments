import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt

def get_func_segmentation(subjectID, resolution, sessionID, runID, plot=False):
    """
    Get head segmentation in fMRI space for the specified NSD subject, session 
    and run. The index-to-structure mapping is as follows:

    0: background (air)
    1: scalp and skull
    2: csf
    3: grey matter
    4: white matter

    Parameters
    ----------
    subjectID : str ranging from '01' to '08'
    sessionID : str ranging from '21' to '30'
    runID : str ranging from '01' to '12'
    plot : bool, optional
        Whether to plot a slice of the segmentation. The default is False.

    Returns
    -------
    seg : 3D numpy array of shape (x, y, z)
    """

    path = f"data/sub{subjectID}/func/fmri/sess{sessionID}/run{runID}/"
    fmri_file = f"sub-{subjectID}_ses-nsd{sessionID}_task-nsdcore_run-{runID}_bold"
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

def get_anat_segmentation(subjectID, resolution, plot=False):
    """
    Get head segmentation in anatomical space for the specified NSD subject and 
    for a given resolution. The index-to-structure mapping is as follows:
    
    0: background (air)
    1: scalp and skull
    2: csf
    3: grey matter
    4: white matter

    Parameters
    ----------
    subjectID : str ranging from '01' to '08'
    resolution : str
    plot : bool, optional
        Whether to plot a slice of the segmentation. The default is False.

    Returns
    -------
    seg_data : 3D numpy array of shape (x, y, z)
    seg_path : str
        Path to the head segmentation in anatomical space.
    """

    path     = f"data/sub{subjectID}/anat/"
    T1_file  = f"T1_{resolution}_masked"
    seg_ext = "_head_seg.nii.gz"
    
    if not os.path.isfile(path+T1_file+seg_ext):

        # create directory
        if not os.path.exists(path): os.makedirs(path)
    
        # get anatomical NSD data
        url  = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/subj{subjectID}/anat/"
        head_mask_data = download_nsd_file(url, path, f"brainmask_{resolution}", return_numpy=True)
        brain_seg      = get_brain_seg(url, path, T1_file)

        # create anatomical segmentation 
        seg_data = np.zeros(head_mask_data.shape)
        seg_data[head_mask_data==1] = 1
        seg_data[brain_seg.get_fdata()==1] = 2
        seg_data[brain_seg.get_fdata()==2] = 3
        seg_data[brain_seg.get_fdata()==3] = 4
        seg = nib.Nifti1Image(seg_data, brain_seg.affine, brain_seg.header)
        nib.save(seg, path+T1_file+seg_ext)

    # plot segmentation
    if plot:
        seg_data = nib.load(path+T1_file+seg_ext).get_fdata()
        plt.imshow(seg_data[:,:,seg_data.shape[2]//2]); plt.colorbar()
        plt.title('Segmentation in anatomical space'); plt.show()
    
    seg_path = path+T1_file+seg_ext
    return seg_data, seg_path

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