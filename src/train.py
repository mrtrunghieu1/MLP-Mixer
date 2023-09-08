import tensorflow as tf
from keras import layers
from tensofflow.keras.layers import Dense, LayerNormalization


class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, S, C, DS, DC):
        super(MLPBlock, self).__init__()
        self.layerNorm1 = LayerNormalization()
        self.layerNorm2 = LayerNormalization()
        w_init = tf.random_normal_initializer()
        self.DS = DS
        self.DC = DC

        self.W1 = tf.Variable(
            initial_value=w_init(shape=(S, DS), dtype="float32"),
            trainable=True
        )
        self.W2 = tf.Variable(
            initial_value=w_init(shape=(DS, S), dtype="float32")
        )

        self.W3 = tf.Variable(
            initial_value=w_init(shape=(C, DC), dtype="float32")
        )

        self.W4 = tf.Variable(
            initial_value=w_init(shape=(DC, C), dtype="float32")
        )

    def call(self, X):
        """
        :param X: (bach_size, S, C)
        :return:
        """
        # patches (batch_size, S, C)
        batch_size, S, C = X.shape

        # Token-mixing MLPs
        # (batch_size, S, C) ==> (batch_size, C, S)
        X_T = tf.transpose(self.layerNorm1(X), perm=(0, 2, 1))
        assert X_T.shape == (batch_size, C, S), f"X_T.shape: {X_T.shape}"

        W1X = tf.matmul(X_T, self.W1)  # (batch_size, C, S) . (S, DS) = (batch_size, C, DS)

        # Equation 1 in this paper (Token-mixing MLPS)
        # (batch_size, C, DS) . (DS, S) = (batch_size, C, S)
        # (batch_size, C, S).T = (batch_size, S, C)
        # (batch_size, S, C) + (batch_size, S, C) = (batch_size, S, C)
        U = tf.transpose(tf.matmul(tf.nn.gelu(W1X), self.W2), perm=(0, 2, 1)) + X  # (1)

        # Equation 2 in this paper (Channel-mixing MLPS)
        W3U = tf.matmul(self.layerNorm2(U), self.W3)  # (batch_size, S, C) . (C, DC) = (batch_size, S, DC)
        # Y =  (batch_size, S, DC) . (DC, C) + (batch_size, S, C)= (batch_size, S, C)
        Y = tf.matmul(tf.nn.gelu(W3U), self.W4) + U
        return Y





class MLPMixer(tf.keras.models.Model):
    def __init__(self, patch_size, S, C, DS, DC, number_of_mlp_blocks):
        """

        :param patch_size:
        :param S:
        :param C:
        :param DS:
        :param DC:
        """
        super(MLPMixer, self).__init__()
        self.projection = Dense(C)
        self.mlp_blocks = [MLPBlock(S, C, DS, DC) for _ in range(number_of_mlp_blocks)]
