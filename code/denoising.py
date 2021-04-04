import argparse
import functools

import tensorflow as tf

import callbacks
import metrics
import skip
import utils


def denoising_train_dataset_generator(x, y, noise_std=1 / 30):
    while True:
        yield (tf.add(x, tf.random.normal(tf.shape(x), stddev=noise_std)), y)


def build_denoising_initial_input(input_img, input_depth, maxval=0.1, data_format=None):
    input_shape = None
    if data_format != "NCHW":
        size = tf.shape(input_img)[tf.rank(input_img) - 3 : tf.rank(input_img) - 1]
        input_shape = (1, size[0], size[1], input_depth)
    else:
        size = tf.shape(input_img)[tf.rank(input_img) - 2 : tf.rank(input_img)]
        input_shape = (1, input_depth, size[0], size[1])
    return tf.random.uniform(input_shape, 0, maxval)


def build_denoising_model(summary=False, plot=None, data_format=None):
    model = skip.build_skip_net(
        32,
        3,
        5,
        128,
        128,
        4,
        upsample_modes="bilinear",
        padding_mode="reflect",
        data_format=data_format,
        activations=functools.partial(tf.keras.layers.LeakyReLU, 0.2),
    )
    if summary:
        model.summary(line_length=150)
    if plot:
        tf.keras.utils.plot_model(
            model, to_file=plot, show_shapes=True, expand_nested=True, dpi=192
        )
    return model


def get_denoising_options():
    parser = argparse.ArgumentParser(description="Denoising model")

    input_group = parser.add_argument_group("input")
    input_group.add_argument("--input", type=str, default=None, help="original image")
    input_group.add_argument("--noisy", type=str, default=None, help="noisy image")

    parser.add_argument("--epochs", type=int, default=3000, help="number of epochs")
    parser.add_argument(
        "--input_depth", type=int, default=32, help="number of features"
    )

    parser.add_argument(
        "--use_nchw",
        action="store_true",
        help="whether to use `NCHW` or `NHWC` (default `NHWC`)",
    )

    parser.add_argument(
        "--rootdir",
        type=str,
        default=None,
        help="location to save model outputs after some epochs",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=100,
        help="how many epochs between each saving model outputs",
    )

    parser.add_argument(
        "--denoised", type=str, default=None, help="where to save the denoised image"
    )

    args = parser.parse_args()
    if not (args.input or args.noisy):
        parser.error("No action requested, add --input or --noisy")

    return args


if __name__ == "__main__":
    args = get_denoising_options()

    data_format = "NCHW" if args.use_nchw else "NHWC"

    input_img, noisy_img = None, None
    if args.input is not None:
        input_img = utils.load_img(args.input, data_format=data_format)
    if args.noisy is not None:
        noisy_img = utils.load_img(args.noisy, data_format=data_format)
        noisy_img = tf.expand_dims(noisy_img, axis=0)
    else:
        input_img = tf.expand_dims(input_img, axis=0)
        noisy_img = utils.make_noisy_img(input_img)

    input_net = build_denoising_initial_input(
        noisy_img, args.input_depth, data_format=data_format
    )

    model = build_denoising_model(data_format=data_format)
    addition_imgs = {"original": input_img} if input_img is not None else {}
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=metrics.build_psnr_metrics(addition_imgs=addition_imgs),
    )

    save_predictions_callback = callbacks.SavePredictions(
        input_net, step=args.step, rootdir=args.rootdir
    )
    model.fit(
        x=denoising_train_dataset_generator(input_net, noisy_img),
        epochs=args.epochs,
        steps_per_epoch=1,
        callbacks=[save_predictions_callback],
        verbose=2,
    )
    denoised_img = model(input_net)
    if args.denoised is not None:
        utils.save_img(args.denoised, denoised_img, data_format=data_format)
