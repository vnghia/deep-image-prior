import functools

import numpy as np
import tensorflow as tf

import skip
import utils


def build_metrics_psnr_original(original):
    def psnr_original(_, y_pred):
        return tf.image.psnr(original, y_pred, max_val=1.0)

    return psnr_original


def denoising_input_generator(net_input, noisy, noise_std):
    while True:
        yield (
            tf.add(net_input, tf.random.normal(net_input.shape, stddev=noise_std)),
            noisy,
        )


def build_denoising_model(original, summary=True, plot=False):
    model = skip.build_skip_net(
        32,
        3,
        5,
        128,
        128,
        4,
        upsample_modes="bilinear",
        padding_mode="reflect",
        activations=functools.partial(tf.keras.layers.LeakyReLU, 0.2),
    )
    if summary:
        model.summary(line_length=150)
    if plot:
        tf.keras.utils.plot_model(
            model,
            to_file="model.png",
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=192,
        )
    PSNR = functools.partial(tf.image.psnr, max_val=1.0)
    functools.update_wrapper(PSNR, tf.image.psnr)
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=[PSNR, build_metrics_psnr_original(original)],
    )
    return model


def train_denoising_model(model, net_input, noisy, epochs):
    model.fit(
        x=denoising_input_generator(net_input, noisy, 1 / 30),
        epochs=epochs,
        batch_size=1,
        steps_per_epoch=1,
        callbacks=[utils.PredictionCallback(net_input)],
    )
    return model


if __name__ == "__main__":
    original = utils.imread_float("res/denoising/input.png")
    noisy = utils.get_noisy_image(original, 25 / 255)
    original = np.expand_dims(original, axis=0)
    noisy = np.expand_dims(noisy, axis=0)
    model = build_denoising_model(original, summary=False, plot=True)
    net_input = tf.random.uniform(
        shape=(1, original.shape[1], original.shape[2], 32), maxval=1 / 10
    )
    model = train_denoising_model(model, net_input, noisy, 3000)
