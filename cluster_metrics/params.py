

class QualityMetricsParams():
    
    
    def __init__(self, isi_threshold= None, min_isi = None , tbin_sec = None):
        if isi_threshold is None:
            self.isi_threshold = 0.0015      # help='Maximum time (in seconds) for ISI violation')
        else: 
            self.isi_threshold = isi_threshold    # help='Maximum time (in seconds) for ISI violation')
        if min_isi is None:
            self.min_isi = 0.00              # help='Minimum time (in seconds) for ISI violation')
        else: 
            self.min_isi = min_isi              # help='Minimum time (in seconds) for ISI violation')
        if tbin_sec is None:
            self.tbin_sec = 0.001            # help='time bin in seconds for ccg in contam_rate calculation')
        else:
            self.tbin_sec = tbin_sec           # help='time bin in seconds for ccg in contam_rate calculation')

