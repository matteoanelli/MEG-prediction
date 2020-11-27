class SPoC_params(object):
    def __init__(self, subject_n, finger, duration, overlap):
        self.subject_n = subject_n
        self.finger = finger
        self.duration = duration
        self.overlap = overlap

    def __str__(self):
        return "_" + str(self.subject_n) + "_" + str(self.finger) + "_" + str(self.duration) + "_" + str(self.overlap)
