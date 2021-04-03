import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def imread_float(fname):
    return mpimg.imread(fname)


def plot_image(image, figsize=(4, 2)):
    plt.figure(figsize=figsize)
    if len(image.shape) == 4:
        image = np.squeeze(image, axis=0)
    plt.imshow(image)
    plt.show()


def plot_images(images, rows=1, cols=1, figsize=(4, 2)):
    _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    for ind, image in enumerate(images):
        if len(image.shape) == 4:
            image = np.squeeze(image, axis=0)
        ax.ravel()[ind].imshow(image)
        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()


def get_noisy_image(image, sigma):
    return np.clip(np.add(image, np.random.normal(scale=sigma, size=image.shape)), 0, 1)


def get_noise(input_shape, var=0.1):
    return tf.random.uniform(shape=input_shape, maxval=var)
