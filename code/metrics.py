import tensorflow as tf


def build_psnr_metrics(with_y_true=True, addition_imgs=None):
    metrics = []
    if with_y_true:

        def psnr(y_true, y_pred):
            return tf.image.psnr(y_true, y_pred, max_val=1.0)

        metrics.append(psnr)

    for name, img in addition_imgs.items():
        if tf.rank(img) == 3:
            img = tf.expand_dims(img, axis=0)

        def psnr(_, y_pred):
            return tf.image.psnr(img, y_pred, max_val=1.0)

        psnr.__name__ = f"psnr_{name}"

        metrics.append(psnr)

    return metrics
