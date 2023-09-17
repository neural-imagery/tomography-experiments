

class Geometry(object):
    """
    This class defines the source and detector geometry for a NSD subject.
    """

    def __init__(self, sources, detectors, directions):
        self.srcpos    = sources
        self.detpos  = detectors
        self.srcdirs = directions
        self.nsources   = sources.shape[0]
        self.ndetectors = detectors.shape[0]

