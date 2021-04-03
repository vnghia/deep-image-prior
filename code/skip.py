import tensorflow as tf

import layers


def build_skip_net(
    input_channels=2,
    output_channels=3,
    levels=5,
    down_channels=128,
    up_channels=128,
    skip_channels=4,
    down_sizes=3,
    up_sizes=3,
    skip_sizes=1,
    downsample_modes="stride",
    upsample_modes="nearest",
    padding_mode="zero",
    use_sigmoid=True,
    use_bias=True,
    use_1x1up=True,
    activations=tf.keras.layers.LeakyReLU,
):
    """Build an autoencoder network with skip connections.

    Arguments:
        input_channels: Integer
        output_channels: Integer
        levels: Integer. Number of encoder and decoder pairs
        down_channels: An integer or tuple/list of integers. How many channels
            in each encoder.
        up_channels: An integer or tuple/list of integers. How many channels
            in each decoder.
        skip_channels: An integer or tuple/list of integers. How many channels
            in each skip connection.
        down_sizes: An integer or tuple/list of integers. `kernel_size` of
            each encoder.
        up_sizes: An integer or tuple/list of integers. `kernel_size` of
            each decoder.
        skip_sizes: An integer or tuple/list of integers. `kernel_size` of
            each skip connection.
        downsample_modes: A string or tuple/list of strings. One of `"stride"`.
        upsample_modes: A string or tuple/list of strings. One of `"nearest"`
            or `"bilinear"`.
        padding_mode: One of `"constant"`, `"reflect"`, `"symmetric"`,
            or `"zero"` (case-insensitive).
        use_sigmoid: Boolean
        use_bias: Boolean, whether the layer uses a bias vector.
        use_1x1up: Boolean
        activations: Activation function to use.

    Returns:
        A `tf.keras.Model`.

    """

    down_channels = tf.constant(down_channels, shape=levels)
    up_channels = tf.constant(up_channels, shape=levels)
    skip_channels = tf.constant(skip_channels, shape=levels)

    down_sizes = tf.constant(down_sizes, shape=levels)
    up_sizes = tf.constant(up_sizes, shape=levels)
    skip_sizes = tf.constant(skip_sizes, shape=levels)

    downsample_modes = (
        tf.constant(downsample_modes, shape=levels, dtype=tf.string).numpy().astype(str)
    )
    upsample_modes = (
        tf.constant(upsample_modes, shape=levels, dtype=tf.string).numpy().astype(str)
    )

    inputs = tf.keras.layers.Input(shape=(None, None, input_channels))

    # First, we add layers along the deeper branch.
    deeper_startnodes = [None] * (levels + 1)
    deeper_startnodes[0] = inputs

    for i in range(levels):
        with tf.name_scope(f"deeper_{i}"):
            output = layers.ConvWithPad2D(
                down_channels[i],
                down_sizes[i],
                strides=2,
                padding_mode=padding_mode,
                use_bias=use_bias,
            )(deeper_startnodes[i])
            output = tf.keras.layers.BatchNormalization()(output)
            output = activations()(output)

            output = layers.ConvWithPad2D(
                down_channels[i],
                down_sizes[i],
                padding_mode=padding_mode,
                use_bias=use_bias,
            )(output)
            output = tf.keras.layers.BatchNormalization()(output)
            deeper_startnodes[i + 1] = activations()(output)

    # Second, we add skip connections (if any) to deeper main nodes.
    skip_nodes = [None] * (levels)

    for i in range(levels):
        with tf.name_scope(f"skip_{i}"):
            if skip_channels[i] != 0:
                output = layers.ConvWithPad2D(
                    skip_channels[i],
                    skip_sizes[i],
                    padding_mode=padding_mode,
                    use_bias=use_bias,
                )(deeper_startnodes[i])
                output = tf.keras.layers.BatchNormalization()(output)
                skip_nodes[i] = activations()(output)

    # Finally, we concat skip connections and deeper (if any) or append
    # deeper (if there's no skip connections). Note that some final layers of each deeper
    # has to be connected with the ending node of the sublayers. Therefore in this loop,
    # `deeper_endnodes` will be built from the deepest layer first.

    deeper_endnodes = [None] * (levels + 1)
    deeper_endnodes[-1] = deeper_startnodes[-1]

    # Reversed loop because we build the deepest layer first.
    for i in range(levels - 1, -1, -1):
        with tf.name_scope(f"deeper_{i}"):
            # Upsampling before concatenation.
            deeper_endnodes[i + 1] = tf.keras.layers.UpSampling2D(
                interpolation=upsample_modes[i]
            )(deeper_endnodes[i + 1])

            output = (
                tf.keras.layers.Concatenate(axis=-1)(
                    [skip_nodes[i], deeper_endnodes[i + 1]]
                )
                if skip_channels[i] != 0
                else deeper_endnodes[i + 1]
            )

            # Some final layers for deeper.
            output = tf.keras.layers.BatchNormalization()(output)

            output = layers.ConvWithPad2D(
                up_channels[i],
                up_sizes[i],
                padding_mode=padding_mode,
                use_bias=use_bias,
            )(output)
            output = tf.keras.layers.BatchNormalization()(output)
            output = activations()(output)

            if use_1x1up:
                output = layers.ConvWithPad2D(
                    up_channels[i], 1, padding_mode=padding_mode, use_bias=use_bias
                )(output)
                output = tf.keras.layers.BatchNormalization()(output)
                output = activations()(output)

            deeper_endnodes[i] = output

    # Final touches
    outputs = layers.ConvWithPad2D(
        output_channels, 1, padding_mode=padding_mode, use_bias=use_bias
    )(deeper_endnodes[0])
    if use_sigmoid:
        outputs = tf.keras.layers.Activation(tf.nn.sigmoid)(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
