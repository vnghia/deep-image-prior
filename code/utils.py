import logging

import matplotlib.pyplot as plt
import tensorflow as tf

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def convert_data_format(data_format, to_tf_format=True):
    if not to_tf_format and (data_format == "NHWC" or data_format is None):
        return "channels_last"
    elif not to_tf_format and (data_format == "NCHW" or data_format is None):
        return "channels_first"
    elif to_tf_format and (data_format == "channels_last" or data_format is None):
        return "NHWC"
    elif to_tf_format and (data_format == "channels_first" or data_format is None):
        return "NCHW"
    else:
        return data_format


def load_img(path, data_format=None, dtype=tf.float32):
    data_format = convert_data_format(data_format, False)
    img = tf.keras.preprocessing.image.load_img(path)
    img = tf.keras.preprocessing.image.img_to_array(img, data_format)
    return tf.math.divide(img, 256.0)


def save_img(path, img, data_format=None):
    tf.keras.preprocessing.image.save_img(
        path, img, data_format=convert_data_format(data_format, to_tf_format=False)
    )


def plot_img(imgs, nrows=None, ncols=3, figsize=(5, 5), data_format=None):
    if tf.rank(imgs) == 3:
        imgs = tf.expand_dims(imgs, axis=0)
    if convert_data_format(data_format) == "NCHW":
        imgs = tf.transpose(imgs, [0, 2, 3, 1])
    nimgs = tf.shape(imgs)[0]
    nrows = int(tf.math.ceil(nimgs / ncols)) if nrows is None else nrows
    _, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for i, img in enumerate(imgs):
        axes.ravel()[i].imshow(img)
        axes.ravel()[i].set_axis_off()
        axes.ravel()[i].set_aspect(1)
    plt.tight_layout()
    plt.show()


def make_noisy_img(img, noise_std=25 / 255):
    return tf.clip_by_value(
        tf.add(img, tf.random.normal(tf.shape(img), stddev=noise_std)), 0, 1
    )
