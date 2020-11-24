"""
    Parameters classes used mainly for the SCNN tuning process. The are also logged using mlflow traker.
    TODO: Different classes for each models.
"""

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


class Params_tunable(object):
    def __init__(self, subject_n, hand, batch_size, valid_batch_size, test_batch_size, epochs, lr, duration, overlap,
                 patience, device, y_measure, s_n_layer, s_kernel_size, t_n_layer, t_kernel_size, max_pooling,
                 ff_n_layer, ff_hidden_channels, dropout, activation):

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
