import tensorflow as tf


class Embedding(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        contraint=None,
        regularizer=None,
        initializer=None,
        **kwargs,
    ):
        super(Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.contraint = tf.keras.constraints.get(contraint)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            dtype=tf.float32,
            shape=[self.vocab_size, self.embed_dim],
            initializer=self.initializer,
            trainable=True,
            regularizer=self.regularizer,
            constraint=self.contraint,
        )
        self.built = True

    def call(self, inputs):
        outputs = tf.cast(inputs, dtype=tf.int32)
        return tf.nn.embedding_lookup(self.embeddings, outputs)

    def recognize_tflite(self, inputs):
        outputs = tf.cast(tf.expand_dims(inputs, axis=-1), dtype=tf.int32)
        return tf.gather_nd(self.embeddings, outputs)  # https://github.com/tensorflow/tensorflow/issues/42410

    def get_config(self):
        conf = super(Embedding, self).get_config()
        conf.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "contraint": self.contraint,
                "regularizer": self.regularizer,
                "initializer": self.initializer,
            }
        )
        return conf