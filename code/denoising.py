import functools

import tensorflow as tf

import callbacks
import metrics
import skip
import utils


def denoising_train_dataset_generator(x, y, noise_std=1 / 30):
    while True:
        yield (tf.add(x, tf.random.normal(tf.shape(x), stddev=noise_std)), y)


def build_denoising_initial_input(input_img, input_depth, maxval=0.1):
    size = tf.shape(input_img)[tf.rank(input_img) - 3 : tf.rank(input_img) - 1]
    return tf.random.uniform((1, size[0], size[1], input_depth), 0, maxval)


def build_denoising_model(summary=False, plot=None):
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
            model, to_file=plot, show_shapes=True, expand_nested=True, dpi=192
        )
    return model


if __name__ == "__main__":
    input_img = utils.load_img("res/denoising/input.png")
    input_img = tf.expand_dims(input_img, axis=0)
    noisy_img = utils.make_noisy_img(input_img)
    input_net = build_denoising_initial_input(input_img, 32)

    model = build_denoising_model()
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=metrics.build_psnr_metrics(addition_imgs={"original": input_img}),
    )

    save_predictions_callback = callbacks.SavePredictions(
        input_net, rootdir="res/denoising/train/"
    )
    model.fit(
        x=denoising_train_dataset_generator(input_net, noisy_img),
        epochs=3000,
        steps_per_epoch=1,
        callbacks=[save_predictions_callback],
        verbose=2,
    )
