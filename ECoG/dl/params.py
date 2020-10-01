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

class Params_tunable(object):
    def __init__(self, subject_n, finger, batch_size, valid_batch_size, test_batch_size, epochs, lr, duration, overlap,
                 patience, device, y_measure, s_n_layer, s_kernel_size, t_n_layer, t_kernel_size, max_pooling,
                 ff_n_layer, ff_hidden_channels, dropout, activation):

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
        self.y_measure = y_measure
        self.s_n_layer = s_n_layer
        self.s_kernel_size = s_kernel_size
        self.t_n_layer = t_n_layer
        self.t_kernel_size = t_kernel_size
        self.max_pooling = max_pooling
        self.ff_n_layer = ff_n_layer
        self.ff_hidden_channels = ff_hidden_channels
        self.dropout = dropout
        self.activation = activation

    def __str__(self):
        return "_" + str(self.subject_n) + \
               "_" + str(self.finger) + \
               "_" + str(self.batch_size) + \
               "_" + str(self.epochs) + \
               "_" + str(self.lr) + \
               "_" + str(self.duration) + \
               "_" + str(self.overlap) + \
               "_" + str(self.device) + \
               "_" + str(self.y_measure)
