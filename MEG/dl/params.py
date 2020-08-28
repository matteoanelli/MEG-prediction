class Params(object):
    def __init__(self, batch_size, valid_batch_size, test_batch_size, epochs, lr, duration, overlap, patient, device):
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.duration = duration
        self.overlap = overlap
        self.patient = patient
        self.device = device
