import numpy as np
import tensorflow as tf
import utils


class PredictionCallback(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, net_input, step=100, show_logs=False):
        super().__init__()
        if len(net_input.shape) == 3:
            net_input = np.expand_dims(net_input, axis=0)
        self.net_input = net_input
        self.outputs = []
        self.step = step
        self.show_logs = show_logs

    def on_train_begin(self, logs=None):
        if self.show_logs:
            super().on_train_begin(logs)

    def on_train_end(self, logs=None):
        if self.show_logs:
            super().on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        if self.show_logs:
            super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.show_logs:
            super().on_epoch_end(epoch, logs)
        if epoch % self.step != 0:
            return
        self.outputs.append(np.squeeze(self.model(self.net_input), axis=0))
        utils.plot_image(self.outputs[-1], figsize=(10, 10))

    def on_train_batch_begin(self, batch, logs=None):
        if self.show_logs:
            super().on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        if self.show_logs:
            super().on_train_batch_end(batch, logs)
