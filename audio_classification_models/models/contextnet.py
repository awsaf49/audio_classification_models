from typing import List
import tensorflow as tf

from ..utils import math_util, weights

L2 = tf.keras.regularizers.l2(1e-6)
URL = "https://github.com/awsaf49/audio_classification_models/releases/download/v1.0.8/contextnet.h5"
BLOCKS =[{'nlayers': 1, 'kernel_size': 5, 'filters': 256, 'strides': 1, 'residual': False, 'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 256,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 256,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 256,'strides': 2,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 256,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 256,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 256,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 256,'strides': 2,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 256,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 256,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 256,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 512,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 512,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 512,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 512,'strides': 2,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 512,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 512,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 512,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 512,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 512,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 512,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 5,'kernel_size': 5,'filters': 512,'strides': 1,'residual': True,'activation': 'silu'},
         {'nlayers': 1,'kernel_size': 5,'filters': 640,'strides': 1,'residual': False,'activation': 'silu'}]

def get_activation(
    activation: str = "silu",
):
    activation = activation.lower()
    if activation in ["silu", "swish"]:
        return tf.nn.swish
    elif activation == 'selu':
        return tf.nn.selu
    elif activation == "relu":
        return tf.nn.relu
    elif activation == "linear":
        return tf.keras.activations.linear
    else:
        raise ValueError("activation must be either 'silu', 'swish', 'selu', 'relu' or 'linear'")


class Reshape(tf.keras.layers.Layer):
    def call(self, inputs):
        return math_util.merge_two_last_dims(inputs)


class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_size: int = 3,
        strides: int = 1,
        filters: int = 256,
        activation: str = "silu",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super(ConvModule, self).__init__(**kwargs)
        self.strides = strides
        self.conv = tf.keras.layers.SeparableConv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            depthwise_regularizer=kernel_regularizer,
            pointwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_conv",
        )
        self.bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn")
        self.activation = get_activation(activation)

    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        outputs = self.conv(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.activation(outputs)
        return outputs


class SEModule(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_size: int = 3,
        strides: int = 1,
        filters: int = 256,
        activation: str = "silu",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super(SEModule, self).__init__(**kwargs)
        self.conv = ConvModule(
            kernel_size=kernel_size,
            strides=strides,
            filters=filters,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_conv_module",
        )
        self.activation = get_activation(activation)
        self.fc1 = tf.keras.layers.Dense(filters // 8, name=f"{self.name}_fc1")
        self.fc2 = tf.keras.layers.Dense(filters, name=f"{self.name}_fc2")

    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        features, input_length = inputs
        outputs = self.conv(features, training=training)

        se = tf.divide(tf.reduce_sum(outputs, axis=1), tf.expand_dims(tf.cast(input_length, dtype=outputs.dtype), axis=1))
        se = self.fc1(se, training=training)
        se = self.activation(se)
        se = self.fc2(se, training=training)
        se = self.activation(se)
        se = tf.nn.sigmoid(se)
        se = tf.expand_dims(se, axis=1)

        outputs = tf.multiply(outputs, se)
        return outputs


class ConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        nlayers: int = 3,
        kernel_size: int = 3,
        filters: int = 256,
        strides: int = 1,
        residual: bool = True,
        activation: str = "silu",
        alpha: float = 1.0,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super(ConvBlock, self).__init__(**kwargs)

        self.dmodel = filters
        self.time_reduction_factor = strides
        filters = int(filters * alpha)

        self.convs = []
        for i in range(nlayers - 1):
            self.convs.append(
                ConvModule(
                    kernel_size=kernel_size,
                    strides=1,
                    filters=filters,
                    activation=activation,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    name=f"{self.name}_conv_module_{i}",
                )
            )

        self.last_conv = ConvModule(
            kernel_size=kernel_size,
            strides=strides,
            filters=filters,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_conv_module_{nlayers - 1}",
        )

        self.se = SEModule(
            kernel_size=kernel_size,
            strides=1,
            filters=filters,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_se",
        )

        self.residual = None
        if residual:
            self.residual = ConvModule(
                kernel_size=kernel_size,
                strides=strides,
                filters=filters,
                activation="linear",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{self.name}_residual",
            )

        self.activation = get_activation(activation)

    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        features, input_length = inputs
        outputs = features
        for conv in self.convs:
            outputs = conv(outputs, training=training)
        outputs = self.last_conv(outputs, training=training)
        input_length = math_util.get_reduced_length(input_length, self.last_conv.strides)
        outputs = self.se([outputs, input_length], training=training)
        if self.residual is not None:
            res = self.residual(features, training=training)
            outputs = tf.add(outputs, res)
        outputs = self.activation(outputs)
        return outputs, input_length


class ContextNetEncoder(tf.keras.Model):
    def __init__(
        self,
        blocks: List[dict] = BLOCKS,
        alpha: float = 0.5,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name='contextnet_encoder',
        **kwargs,
    ):
        super(ContextNetEncoder, self).__init__(name=name, **kwargs)

        self.reshape = Reshape(name=f"{self.name}_reshape")

        self.blocks = []
        for i, config in enumerate(blocks):
            self.blocks.append(
                ConvBlock(
                    **config,
                    alpha=alpha,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    name=f"{self.name}_block_{i}",
                )
            )

    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        outputs = inputs # shape: [B, T, F, C]
        input_length  = tf.expand_dims(tf.shape(inputs)[1], axis=0) # spec time duration
        outputs = self.reshape(outputs)
        for block in self.blocks:
            outputs, input_length = block([outputs, input_length], training=training)
        return outputs

def ContextNet(input_shape=(128, 80, 1), num_classes=1, final_activation='sigmoid', pretrain=True):
    inp = tf.keras.layers.Input(shape=input_shape)
    backbone = ContextNetEncoder()
    out = backbone(inp)
    if pretrain:
        weights.load_pretrain(backbone, url=URL)
    out = tf.keras.layers.GlobalAveragePooling1D()(out)
    out = tf.keras.layers.Dense(32, activation='selu')(out)
    out = tf.keras.layers.Dense(num_classes, activation=final_activation)(out)
    model = tf.keras.models.Model(inp, out)
    return model