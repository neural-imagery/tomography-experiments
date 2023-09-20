import os
import numpy as np

class Geometry(object):
    """
    This class defines the source and detector geometry for a NSD subject.
    """

    def __init__(self, sources, detectors, directions):
        self.sources    = sources
        self.detectors  = detectors
        self.directions = directions
        self.nsources   = sources.shape[0]
        self.ndetectors = detectors.shape[0]

    def initialize_src_dets(self, path):
        """
        Creates source and closest detector subsets.
        """
        src_dets_path = path + "detector_positions/"
        os.makedirs(src_dets_path)
        for src_idx in range(self.nsources):
            detectors = self.get_closest_detectors(src_idx)
            np.save(src_dets_path + f"detectors_src{src_idx}.npy", detectors)


    def get_closest_detectors(self, src_idx, maxdist=60, scaling=1.8):
        """
        Restrict detectors to those closest to a given source.

        Parameters
        ----------
        source_idx : int
        maxdist : float (in mm)

        Returns
        -------
        detectors : np.ndarray
            Position of detectors
        """
        srcpos = self.sources[src_idx]
        detpos = self.detectors[:, :3]

        # compute distance between source and detectors
        dist = np.linalg.norm((srcpos - detpos)*scaling, axis=1)

        # keep only the closest detectors
        detector_idx = np.where(dist < maxdist)[0]
        newdetpos = detpos[detector_idx]

        # update geom (while keeping the original radius stored in the 4th column)
        detectors = np.zeros((newdetpos.shape[0], 4))
        detectors[:, :3] = newdetpos
        detectors[:, 3] = self.detectors[detector_idx, 3]

        return detectors
