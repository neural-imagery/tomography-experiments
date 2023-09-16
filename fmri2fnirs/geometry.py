

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

