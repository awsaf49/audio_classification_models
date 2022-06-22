import tensorflow as tf


class GLU(tf.keras.layers.Layer):
    def __init__(
        self,
        axis=-1,
        name="glu_activation",
        **kwargs,
    ):
        super(GLU, self).__init__(name=name, **kwargs)
        self.axis = axis

    def call(
        self,
        inputs,
        **kwargs,
    ):
        a, b = tf.split(inputs, 2, axis=self.axis)
        b = tf.nn.sigmoid(b)
        return tf.multiply(a, b)

    def get_config(self):
        conf = super(GLU, self).get_config()
        conf.update({"axis": self.axis})
        return conf