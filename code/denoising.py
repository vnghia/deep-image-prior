import functools

import numpy as np
import tensorflow as tf

import skip
import utils


def denoising_psnr_original(original):
    def metrics(_, y_pred):
        return tf.image.psnr(y_pred, original, max_val=1.0)

    return metrics


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
        metrics=[PSNR, denoising_psnr_original(original)],
    )
    return model


def train_denoising_model(model, net_input, noisy, epochs):
    model.fit(
        x=net_input,
        y=noisy,
        epochs=epochs,
        batch_size=1,
        steps_per_epoch=1,
        callbacks=[utils.PredictionCallback(net_input)],
    )
    return model


if __name__ == "__main__":
    original = utils.imread_float("res/denoising/input.png")
    original = np.expand_dims(original, axis=0)
    noisy = utils.imread_float("res/denoising/noisy.png")
    noisy = np.expand_dims(noisy, axis=0)
    model = build_denoising_model(original, summary=False, plot=True)
    net_input = utils.get_noise(
        input_shape=(1, original.shape[1], original.shape[2], 32)
    )
    model = train_denoising_model(model, net_input, noisy, 3000)
