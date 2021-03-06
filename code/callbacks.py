import os

import tensorflow as tf

import utils


class SavePredictions(tf.keras.callbacks.Callback):
    def __init__(
        self,
        input_net,
        step=100,
        rootdir=None,
        plot=False,
        data_format=None,
        keep_output=False,
    ):
        super().__init__()
        self.input_net = input_net
        self.outputs_net = []
        self.step = step
        self.rootdir = rootdir
        self.plot = plot
        self.data_format = data_format
        self.keep_output = keep_output

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.step != 0:
            return
        result = tf.squeeze(self.model(self.input_net), axis=0)
        if self.rootdir is not None:
            utils.save_img(
                os.path.join(self.rootdir, f"epoch_{epoch}.png"),
                result,
                self.data_format,
            )
        if self.keep_output:
            self.outputs_net.append(result)
        if self.plot:
            utils.plot_img(result, ncols=1, data_format=self.data_format)
