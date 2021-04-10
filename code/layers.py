import tensorflow as tf

import utils


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
        if mode.lower() not in ("constant", "reflect", "symmetric", "zero"):
            raise ValueError(f"mode {mode} is invalid")
        self.mode = mode.lower()
        if self.mode == "zero":
            self.mode = "constant"
            self.constant_values = 0
        else:
            self.constant_values = constant_values

    def call(self, inputs):
        return tf.pad(inputs, self.paddings, self.mode, self.constant_values)


class ConvWithPad2D(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding_mode="zero",
        data_format="NHWC",
        use_bias=True,
        **kwargs,
    ):
        super().__init__(trainable=True, **kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.strides = strides
        self.padding_mode = str(padding_mode.lower())
        if self.padding_mode == "zero":
            self.padding_mode = "constant"
        self.data_format = str(utils.convert_data_format(data_format))
        self.use_bias = bool(use_bias)

    def get_config(self):
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding_mode": self.padding_mode,
            "data_format": self.data_format,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        # Padding
        self.paddings = tf.constant((self.kernel_size - 1) // 2, shape=[2, 2])
        self.paddings = tf.concat([[[0, 0]], self.paddings], 0)  # `batch_size`
        if self.data_format == "NHWC":
            self.paddings = tf.concat([self.paddings, [[0, 0]]], 0)  # `channels_last`
        elif self.data_format == "NCHW":
            self.paddings = tf.concat([[[0, 0]], self.paddings], 0)  # `channels_first`
        # End padding

        in_channels = None
        if self.data_format == "NHWC":
            in_channels = input_shape[-1]
        elif self.data_format == "NCHW":
            in_channels = input_shape[1]
        else:
            raise ValueError(f"data_format {self.data_format} is invalid")
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_size, self.kernel_size, in_channels, self.filters),
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                trainable=True,
            )

    def call(self, inputs):
        outputs = tf.pad(inputs, self.paddings, self.padding_mode, 0)
        outputs = tf.nn.conv2d(
            outputs,
            self.kernel,
            self.strides,
            padding=[[0, 0], [0, 0], [0, 0], [0, 0]],
            data_format=self.data_format,
        )
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, self.data_format)
        return outputs
