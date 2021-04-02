import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging

from tensorflow.python.ops.gen_math_ops import asin

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def imread_float(fname):
    return mpimg.imread(fname)


def plot_image(image):
    if len(image.shape) == 4:
        image = np.squeeze(image, axis=0)
    plt.imshow(image)
    plt.show()


def get_noise(input_shape, var=0.1):
    return tf.random.uniform(shape=input_shape, maxval=var)


class PredictionCallback(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, net_input, step=100):
        super().__init__()
        if len(net_input.shape) == 3:
            net_input = np.expand_dims(net_input, axis=0)
        self.net_input = net_input
        self.outputs = []
        self.step = step

    def on_epoch_end(self, epoch, logs):
        super().on_epoch_end(epoch, logs)
        if epoch % self.step != 0:
            return
        self.outputs.append(np.squeeze(self.model(self.net_input), axis=0))
        plot_image(self.outputs[-1])
