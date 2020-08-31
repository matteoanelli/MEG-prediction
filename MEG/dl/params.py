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

    def __str__(self):
        return "_" + str(self.batch_size) + \
               "_" + str(self.epochs) + \
               "_" + str(self.lr) + \
               "_" + str(self.duration) + \
               "_" + str(self.overlap) + \
               "_" + str(self.device)
