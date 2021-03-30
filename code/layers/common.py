import tensorflow as tf


class Pad2D(tf.keras.layers.Layer):
    """Pads a 2D input tensor.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (`padding_top`,
            `padding_bottom`, `padding_left`, `padding_right`).
        mode: One of `"CONSTANT"`, `"REFLECT"`, `"SYMMETRIC"`, or `"ZERO"` (case-insensitive).
        constant_values: In "CONSTANT" mode, the scalar pad value to use.
        data_format: An optional `string` from: `"NHWC", "NCHW"`.
            Defaults to `"NHWC"`.

    Returns:
        A `Tensor`.

    """

    def __init__(
        self, paddings, mode="CONSTANT", constant_values=0, data_format="NHWC", **kwargs
    ):
        super().__init__(trainable=False, **kwargs)
        self.paddings = tf.constant(paddings, shape=[2, 2])
        self.paddings = tf.concat([[[0, 0]], self.paddings], 0)  # `batch_size`
        if data_format == "NHWC":
            self.paddings = tf.concat([self.paddings, [[0, 0]]], 0)  # `channels_last`
        elif data_format == "NCHW":
            self.paddings = tf.concat([[[0, 0]], self.paddings], 0)  # `channels_first`
        else:
            raise ValueError(f"data_format {data_format} is invalid")
        if mode.capitalize() not in ("CONSTANT", "REFLECT", "SYMMETRIC", "ZERO"):
            raise ValueError(f"mode {mode} is invalid")
        self.mode = mode
        if self.mode.capitalize() == "ZERO":
            self.mode = "CONSTANT"
            self.constant_values = 0
        else:
            self.constant_values = constant_values

    def call(self, inputs):
        return tf.pad(inputs, self.paddings, self.mode, self.constant_values)
