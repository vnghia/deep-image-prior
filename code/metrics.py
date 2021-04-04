import tensorflow as tf

import utils


def build_psnr_metrics(with_y_true=True, addition_imgs=None, data_format=None):
    metrics = []
    if with_y_true:

        def psnr(y_true, y_pred):
            if utils.convert_data_format(data_format) == "NCHW":
                y_true = tf.transpose(y_true, [0, 2, 3, 1])
                y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
            return tf.image.psnr(y_true, y_pred, max_val=1.0)

        metrics.append(psnr)

    if addition_imgs is None:
        return metrics

    for name, img in addition_imgs.items():
        if tf.rank(img) == 3:
            img = tf.expand_dims(img, axis=0)
        if utils.convert_data_format(data_format) == "NCHW":
            img = tf.transpose(img, [0, 2, 3, 1])

        def psnr(_, y_pred):
            if utils.convert_data_format(data_format) == "NCHW":
                y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
            return tf.image.psnr(img, y_pred, max_val=1.0)

        psnr.__name__ = f"psnr_{name}"

        metrics.append(psnr)

    return metrics
