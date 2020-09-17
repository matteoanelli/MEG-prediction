class Params(object):
    def __init__(self, subject_n, hand, batch_size, valid_batch_size, test_batch_size, epochs, lr, duration, overlap,
                 patience, device, y_measure):
        self.subject_n = subject_n
        self.hand = hand
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.duration = duration
        self.overlap = overlap
        self.patience = patience
        self.device = device
        self.y_measure = y_measure

    def __str__(self):
        return "_" + str(self.subject_n) + \
               "_" + str(self.hand) + \
               "_" + str(self.batch_size) + \
               "_" + str(self.epochs) + \
               "_" + str(self.lr) + \
               "_" + str(self.duration) + \
               "_" + str(self.overlap) + \
               "_" + str(self.device) + \
               "_" + str(self.y_measure)

class SPoC_params(object):
    def __init__(self, subject_n, hand, duration, overlap, y_measure):
        self.subject_n = subject_n
        self.hand = hand
        self.duration = duration
        self.overlap = overlap
        self.y_measure = y_measure

    def __str__(self):
        return "_" + str(self.subject_n) + \
               "_" + str(self.hand) + \
               "_" + str(self.duration) + \
               "_" + str(self.overlap) + \
               "_" + str(self.y_measure)
