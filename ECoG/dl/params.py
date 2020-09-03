class Params(object):
    def __init__(self, dataset, subject_n, finger, batch_size, valid_batch_size, test_batch_size, epochs, lr, duration, overlap,
                 patience, device, sampling_rate):
        self.dataset = dataset
        self.subject_n = subject_n
        self.finger = finger
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.duration = duration
        self.overlap = overlap
        self.patience = patience
        self.device = device
        self.sampling_rate = sampling_rate

    def __str__(self):
        return "_" + str(self.dataset) + \
               "_" + str(self.subject_n) + \
               "_" + str(self.finger) + \
               "_" + str(self.batch_size) + \
               "_" + str(self.epochs) + \
               "_" + str(self.lr) + \
               "_" + str(self.duration) + \
               "_" + str(self.overlap) + \
               "_" + str(self.device) + \
               "_" + str(self.sampling_rate)
