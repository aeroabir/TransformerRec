import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.layers import (
    Conv1D,
    Input,
    concatenate,
    Dropout,
    BatchNormalization,
    Reshape,
)
from tensorflow.keras.layers import (
    Flatten,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense,
    Permute,
)
from tensorflow.keras.models import Model


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    - Q (query), K (key) and V (value) are split into multiple heads (num_heads)
    - each tuple (q, k, v) are fed to scaled_dot_product_attention
    - all attention outputs are concatenated
    """

    def __init__(self, attention_dim, num_heads, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        assert attention_dim % self.num_heads == 0
        self.dropout_rate = dropout_rate

        self.depth = attention_dim // self.num_heads

        self.Q = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.K = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.V = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, queries, keys):

        # Linear projections
        Q = self.Q(queries)  # (N, T_q, C)
        K = self.K(keys)  # (N, T_k, C)
        V = self.V(keys)  # (N, T_k, C)

        # --- MULTI HEAD ---
        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # --- SCALED DOT PRODUCT ---
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # print(outputs.shape)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)

        key_masks = tf.tile(key_masks, [self.num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(
            tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]
        )  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-(2 ** 32) + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Future blinding (Causality)
        diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(
            diag_vals
        ).to_dense()  # (T_q, T_k)
        masks = tf.tile(
            tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]
        )  # (h*N, T_q, T_k)

        paddings = tf.ones_like(masks) * (-(2 ** 32) + 1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [self.num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(
            tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]
        )  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # --- MULTI HEAD ---
        # concat heads
        outputs = tf.concat(
            tf.split(outputs, self.num_heads, axis=0), axis=2
        )  # (N, T_q, C)

        # Residual connection
        outputs += queries

        return outputs


class PointWiseFeedForward(tf.keras.layers.Layer):
    """
    Convolution layers with residual connection
    """

    def __init__(self, conv_dims, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv_dims = conv_dims
        self.dropout_rate = dropout_rate
        self.conv_layer1 = tf.keras.layers.Conv1D(
            filters=self.conv_dims[0], kernel_size=1, activation="relu", use_bias=True
        )
        self.conv_layer2 = tf.keras.layers.Conv1D(
            filters=self.conv_dims[1], kernel_size=1, activation=None, use_bias=True
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, x):

        output = self.conv_layer1(x)
        output = self.dropout_layer(output)
        output = self.conv_layer2(output)
        output = self.dropout_layer(output)

        # Residual connection
        output += x

        return output


class EncoderLayer(tf.keras.layers.Layer):
    """
    Transformer based encoder layer

    """

    def __init__(
        self,
        seq_max_len,
        embedding_dim,
        attention_dim,
        num_heads,
        conv_dims,
        dropout_rate,
    ):
        super(EncoderLayer, self).__init__()

        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim

        self.mha = MultiHeadAttention(attention_dim, num_heads, dropout_rate)
        self.ffn = PointWiseFeedForward(conv_dims, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.embedding_dim, 1e-08
        )

    def call_(self, x, training, mask):

        attn_output = self.mha(queries=self.layer_normalization(x), keys=x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # feed forward network
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        # masking
        out2 *= mask

        return out2

    def call(self, x, training, mask):

        x_norm = self.layer_normalization(x)
        attn_output = self.mha(queries=x_norm, keys=x)
        attn_output = self.ffn(attn_output)
        out = attn_output * mask

        return out


class Encoder(tf.keras.layers.Layer):
    """
    Invokes Transformer based encoder with user defined number of layers

    """

    def __init__(
        self,
        num_layers,
        seq_max_len,
        embedding_dim,
        attention_dim,
        num_heads,
        conv_dims,
        dropout_rate,
    ):
        super(Encoder, self).__init__()

        self.num_layers = num_layers

        self.enc_layers = [
            EncoderLayer(
                seq_max_len,
                embedding_dim,
                attention_dim,
                num_heads,
                conv_dims,
                dropout_rate,
            )
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class LayerNormalization(tf.keras.layers.Layer):
    """
    Layer normalization using mean and variance
    gamma and beta are the learnable parameters
    """

    def __init__(self, seq_max_len, embedding_dim, epsilon):
        super(LayerNormalization, self).__init__()
        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.params_shape = (self.seq_max_len, self.embedding_dim)
        g_init = tf.ones_initializer()
        self.gamma = tf.Variable(
            initial_value=g_init(shape=self.params_shape, dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=b_init(shape=self.params_shape, dtype="float32"),
            trainable=True,
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, [-1], keepdims=True)
        normalized = (x - mean) / ((variance + self.epsilon) ** 0.5)
        output = self.gamma * normalized + self.beta
        return output


# https://keras.io/api/models/model/
# Note: the functional API form cannot take additional arguments like training
# for that we have to use the model subclass
def build_multilevel_transformer(item_num, seq_max_len, **kwargs):
    num_blocks = kwargs.get("num_blocks", 2)
    embedding_dim = kwargs.get("embedding_dim", 100)
    attention_dim = kwargs.get("attention_dim", 100)
    attention_num_heads = kwargs.get("attention_num_heads", 1)
    conv_dims = kwargs.get("conv_dims", [100, 100])
    dropout_rate = kwargs.get("dropout_rate", 0.5)
    l2_reg = kwargs.get("l2_reg", 0.0)

    def embedding(input_seq):

        seq_embeddings = item_embedding_layer(input_seq)
        seq_embeddings = seq_embeddings * (embedding_dim ** 0.5)  # should be added?

        # FIXME
        positional_seq = tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0)
        positional_seq = tf.tile(positional_seq, [tf.shape(input_seq)[0], 1])
        positional_embeddings = positional_embedding_layer(positional_seq)

        return seq_embeddings, positional_embeddings

    item_embedding_layer = tf.keras.layers.Embedding(
        item_num + 1,
        embedding_dim,
        name="item_embeddings",
        mask_zero=True,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
    )

    positional_embedding_layer = tf.keras.layers.Embedding(
        seq_max_len,
        embedding_dim,
        name="positional_embeddings",
        mask_zero=False,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
    )
    dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    encoder = Encoder(
        num_blocks,
        seq_max_len,
        embedding_dim,
        attention_dim,
        attention_num_heads,
        conv_dims,
        dropout_rate,
    )
    # mask_layer = tf.keras.layers.Masking(mask_value=0)
    layer_normalization = LayerNormalization(seq_max_len, embedding_dim, 1e-08)
    final = tf.keras.layers.Dense(item_num + 1, activation="linear")

    inp = Input(shape=(seq_max_len,))
    mask = tf.expand_dims(tf.cast(tf.not_equal(inp, 0), tf.float32), -1)
    seq_embeddings, positional_embeddings = embedding(inp)
    seq_embeddings += positional_embeddings
    seq_embeddings = dropout_layer(seq_embeddings)
    seq_embeddings *= mask

    seq_attention = seq_embeddings
    seq_attention = encoder(seq_attention, True, mask)
    seq_attention = layer_normalization(seq_attention)  # (b, s, d)
    seq_attention = tf.reshape(
        seq_attention,
        [tf.shape(seq_attention)[0], seq_max_len * embedding_dim],
    )  # (b, s*d)
    output = final(seq_attention)
    model = Model(inputs=inp, outputs=output)
    return model


class TENCODER(tf.keras.Model):
    """SAS Rec model
    Transformer encoder for sequence to class.

    Args:
        item_num: number of items in the dataset
        seq_max_len: maximum number of items in user history
        num_blocks: number of Transformer blocks to be used
        embedding_dim: item embedding dimension
        attention_dim: Transformer attention dimension
        conv_dims: list of the dimensions of the Feedforward layer
        dropout_rate: dropout rate
        l2_reg: coefficient of the L2 regularization
        num_neg_test: number of negative examples used in testing
    """

    def __init__(self, **kwargs):
        super(TENCODER, self).__init__()

        self.item_num = kwargs.get("item_num", None)
        self.seq_max_len = kwargs.get("seq_max_len", 100)
        self.num_blocks = kwargs.get("num_blocks", 2)
        self.embedding_dim = kwargs.get("embedding_dim", 100)
        self.attention_dim = kwargs.get("attention_dim", 100)
        self.attention_num_heads = kwargs.get("attention_num_heads", 1)
        self.conv_dims = kwargs.get("conv_dims", [100, 100])
        self.dropout_rate = kwargs.get("dropout_rate", 0.5)
        self.l2_reg = kwargs.get("l2_reg", 0.0)

        self.item_embedding_layer = tf.keras.layers.Embedding(
            self.item_num + 1,
            self.embedding_dim,
            name="item_embeddings",
            mask_zero=True,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
        )

        self.positional_embedding_layer = tf.keras.layers.Embedding(
            self.seq_max_len,
            self.embedding_dim,
            name="positional_embeddings",
            mask_zero=False,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.encoder = Encoder(
            self.num_blocks,
            self.seq_max_len,
            self.embedding_dim,
            self.attention_dim,
            self.attention_num_heads,
            self.conv_dims,
            self.dropout_rate,
        )
        self.mask_layer = tf.keras.layers.Masking(mask_value=0)
        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.embedding_dim, 1e-08
        )
        self.dense1 = tf.keras.layers.Dense(self.embedding_dim, activation="relu")
        self.dense_h = tf.keras.layers.Dense(self.embedding_dim, activation="relu")
        self.dense_c = tf.keras.layers.Dense(self.embedding_dim, activation="relu")
        self.decoder_lstm = tf.keras.layers.LSTM(
            self.embedding_dim, activation="tanh", return_sequences=True
        )
        self.final = tf.keras.layers.Dense(self.item_num + 1, activation="linear")

    def embedding(self, input_seq):

        seq_embeddings = self.item_embedding_layer(input_seq)
        seq_embeddings = seq_embeddings * (
            self.embedding_dim ** 0.5
        )  # should be added?

        # FIXME
        positional_seq = tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0)
        positional_seq = tf.tile(positional_seq, [tf.shape(input_seq)[0], 1])
        positional_embeddings = self.positional_embedding_layer(positional_seq)

        return seq_embeddings, positional_embeddings

    def process_session(self, input_seq, training):
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)
        seq_embeddings += positional_embeddings
        seq_embeddings = self.dropout_layer(seq_embeddings)
        seq_embeddings *= mask

        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)
        return seq_attention

    def call(self, x, training):

        seq_attention = self.process_session(x, training)  # (b, s, d)
        seq_attention = tf.reshape(
            seq_attention,
            [tf.shape(x)[0], self.seq_max_len * self.embedding_dim],
        )  # (b, s*d)
        output = self.final(seq_attention)

        return output

    def predict(self, x):
        training = False
        seq_attention = self.process_session(x, training)  # (b, s, d)
        seq_attention = tf.reshape(
            seq_attention,
            [tf.shape(x)[0], self.seq_max_len * self.embedding_dim],
        )  # (b, s*d)

        output = self.final(seq_attention)
        return output

    def loss_function(self, logits, labels):
        # logits: (b, s, V)
        # labels: (b, s)
        # mask: (b, s)
        # print(logits.shape, labels.shape)
        # print(loss_mask.shape)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        loss = loss_object(labels, logits)
        loss = tf.reduce_sum(loss * loss_mask)

        reg_loss = tf.compat.v1.losses.get_regularization_loss()
        loss += reg_loss
        return loss
