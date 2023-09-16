import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt

def get_segmentation(subjectID, resolution, sessionID, runID, plot=False):
    """
    Get head segmentation mask in fMRI space for the specified NSD subject, 
    session and run. The index-to-structure mapping of the mask is as follows:

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
    mask : 3D numpy array of shape (x, y, z)
    """

    path = f"data/sub{subjectID}/func/fmri/sess{sessionID}/run{runID}/"
    fmri_file = f"sub-{subjectID}_ses-nsd{sessionID}_task-nsdcore_run-{runID}_bold"
    mask_ext = "_head_mask"
    ext = ".nii.gz"

    if os.path.isfile(path+fmri_file+mask_ext+".npy"): mask = np.load(path+fmri_file+mask_ext+".npy")
    else: 

        # create directory
        if not os.path.exists(path): os.makedirs(path)    

        # get head segmentation in anatomical space
        download_anat_mask(subjectID, resolution, plot=False)

        # get functional NSD data
        url = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata_rawdata/sub-{subjectID}/ses-nsd{sessionID}/func/"
        download_nsd_file(url, path, fmri_file)

        # TODO: Update anat_mask_file
        # register anatomical segmention in functional space using FSLs `flirt`
        # !flirt -in $anat_mask_file -ref $path$fmri_file$ext -out $path$fmri_file$mask_ext$ext
        anat_mask_file = f"data/sub{subjectID}/anat/T1_{resolution}_masked_head_mask.nii.gz"
        os.system(f"flirt -in {anat_mask_file} -ref {path}{fmri_file}{ext} -out {path}{fmri_file}{mask_ext}{ext}")

        # load and save functional mask
        mask = np.round(nib.load(path+fmri_file+mask_ext+ext).get_fdata()).astype('uint8')
        np.save(path+fmri_file+mask_ext+".npy", mask)

    # plot mask    
    if plot:
        plt.imshow(mask[:,:,mask.shape[2]//2]); plt.colorbar()
        plt.title('Segmented mask in functional space')
        
    return mask

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
        # !wget -P $path $url$file$ext
        os.system(f"wget -P {path} {url}{file}{ext}")
    if return_numpy: return nib.load(path+file+ext).get_fdata()


def get_brain_mask(url, path, T1_file, ext=".nii.gz"):
    """
    Get brain segmentation mask from T1 image.

    Parameters
    ----------
    url : str
    path : str
    T1_file : str

    Returns
    -------
    brain_mask : nibabel.nifti1.Nifti1Image
    """
    
    # get T1 image
    download_nsd_file(url, path, T1_file)
    
    # segment brain from T1 image using FSLs `bet` (~5s)
    brain_ext = "_brain"
    if not os.path.isfile(path+T1_file+brain_ext+ext):
        # !bet $path$T1_file$ext $path$T1_file$brain_ext -m
        os.system(f"bet {path}{T1_file}{ext} {path}{T1_file}{brain_ext} -m")

    # segment csf, grey and white matter from brain using FSLs `fast` (~5m8s)
    if not os.path.isfile(path+T1_file+"_seg"+ext):
        # !fast $path$T1_file$brain_ext
        os.system(f"fast {path}{T1_file}{brain_ext}")

    brain_mask = nib.load(path+T1_file+brain_ext+"_seg"+ext)
    return brain_mask

def download_anat_mask(subjectID, resolution, plot=False):
    """
    Get head segmentation mask in anatomical space for the specified NSD subject.

    Parameters
    ----------
    subjectID : str ranging from '01' to '08'
    resolution : str
    plot : bool, optional
        Whether to plot a slice of the segmentation. The default is False.

    Returns
    -------
    anat_mask_path : str
        Path to the head segmentation mask in anatomical space.
    """

    path     = f"data/sub{subjectID}/anat/"
    T1_file  = f"T1_{resolution}_masked"
    mask_ext = "_head_mask.nii.gz"
    
    if not os.path.isfile(path+T1_file+mask_ext):

        # create directory
        if not os.path.exists(path): os.makedirs(path)
    
        # get anatomical NSD data
        url  = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/subj{subjectID}/anat/"
        head_mask_data = download_nsd_file(url, path, f"brainmask_{resolution}", return_numpy=True)
        brain_mask     = get_brain_mask(url, path, T1_file)

        # create anatomical mask 
        mask_data = np.zeros(head_mask_data.shape)
        mask_data[head_mask_data==1] = 1
        mask_data[brain_mask.get_fdata()==1] = 2
        mask_data[brain_mask.get_fdata()==2] = 3
        mask_data[brain_mask.get_fdata()==3] = 4
        mask = nib.Nifti1Image(mask_data, brain_mask.affine, brain_mask.header)
        nib.save(mask, path+T1_file+mask_ext)

    # plot mask
    if plot:
        mask_data = nib.load(path+T1_file+mask_ext).get_fdata()
        plt.imshow(mask_data[:,:,mask_data.shape[2]//2]); plt.colorbar()
        plt.title('Segmented mask in anatomical space'); plt.show()
    
    mask_path = path+T1_file+mask_ext
    return mask_path