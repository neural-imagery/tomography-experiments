import os
import numpy as np
from fmri2fnirs import get_anatomical_segmentation

class Subject(object):
    
    def __init__(self, id):
        self.id = id

        self.anat_seg = get_anatomical_segmentation(self.id, resolution)
    
    def create_segmentation(self, resolution):

        path     = f"data/sub{self.id}/anat/"
        T1_file  = f"T1_{resolution}_masked"
        seg_ext = "_head_seg.nii.gz"
        
        if not os.path.isfile(path+T1_file+seg_ext):

            # create directory
            if not os.path.exists(path): os.makedirs(path)
        
            # get anatomical NSD data
            url  = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/subj{self.id}/anat/"
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

