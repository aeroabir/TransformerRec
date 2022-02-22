import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys

from .tencoder import LayerNormalization


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
        hidden_dim,
        rnn_name,
        conv_dims,
        dropout_rate,
    ):
        super(EncoderLayer, self).__init__()

        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim

        if rnn_name == "lstm":
            self.rnn = layers.LSTM(embedding_dim, return_sequences=True)
        else:
            self.rnn = layers.GRU(embedding_dim, return_sequences=True)
        # self.mha = MultiHeadAttention(attention_dim, num_heads, dropout_rate)
        self.ffn = PointWiseFeedForward(conv_dims, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.embedding_dim, 1e-08
        )

    # def call_(self, x, training, mask):

    #     attn_output = self.mha(queries=self.layer_normalization(x), keys=x)
    #     attn_output = self.dropout1(attn_output, training=training)
    #     out1 = self.layernorm1(x + attn_output)

    #     # feed forward network
    #     ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    #     ffn_output = self.dropout2(ffn_output, training=training)
    #     out2 = self.layernorm2(
    #         out1 + ffn_output
    #     )  # (batch_size, input_seq_len, d_model)

    #     # masking
    #     out2 *= mask

    #     return out2

    def call(self, x, training, mask):

        x_norm = self.layer_normalization(x)
        attn_output = self.rnn(x_norm)
        attn_output = self.ffn(attn_output)
        out = attn_output * mask

        return out


class Encoder(tf.keras.layers.Layer):
    """
    Invokes RNN based encoder with user defined number of layers

    """

    def __init__(
        self,
        num_layers,
        seq_max_len,
        embedding_dim,
        hidden_dim,
        rnn_name,
        conv_dims,
        dropout_rate,
    ):
        super(Encoder, self).__init__()

        self.num_layers = num_layers

        self.enc_layers = [
            EncoderLayer(
                seq_max_len,
                embedding_dim,
                hidden_dim,
                rnn_name,
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


class RNNREC(tf.keras.Model):
    """RNN Rec model
    RNN Based Sequential Prediction

    Args:
        item_num: number of items in the dataset
        seq_max_len: maximum number of items in user history
        num_blocks: number of RNN blocks to be stacked
        embedding_dim: item embedding dimension
        hidden_dim: RNN hidden dimension
        rnn_name: RNN type, LSTM or GRU
        conv_dims: list of the dimensions of the Feedforward layer
        dropout_rate: dropout rate
        l2_reg: coefficient of the L2 regularization
        num_neg_test: number of negative examples used in testing
    """

    def __init__(self, **kwargs):
        super(RNNREC, self).__init__()

        self.item_num = kwargs.get("item_num", None)
        self.seq_max_len = kwargs.get("seq_max_len", 100)
        self.tgt_seq_len = kwargs.get("tgt_seq_len", 12)
        self.num_blocks = kwargs.get("num_blocks", 2)
        self.embedding_dim = kwargs.get("embedding_dim", 100)
        self.embedding_dims = kwargs.get(
            "embedding_dims", None
        )  # for multiple embeddings
        self.hidden_dim = kwargs.get("hidden_dim", 100)
        self.rnn_name = kwargs.get("rnn_name", "gru")
        self.conv_dims = kwargs.get("conv_dims", [100, 100])
        self.dropout_rate = kwargs.get("dropout_rate", 0.5)
        self.l2_reg = kwargs.get("l2_reg", 0.0)
        self.predict_proba = kwargs.get("predict_proba", False)
        self.multi_feature = kwargs.get("multi_feature", False)

        if self.multi_feature:
            # multiple embeddings for different features
            self.item_embedding_layer = []
            for ii, count in enumerate(self.item_num):
                embedding_ii = tf.keras.layers.Embedding(
                    count + 1,
                    self.embedding_dims[ii],
                    name=f"item_{ii+1}_embeddings",
                    mask_zero=True,
                    embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                )
                self.item_embedding_layer.append(embedding_ii)
            target_vocab_dimension = self.item_num[0] + 1
            self.embedding_out_dim = sum(self.embedding_dims)
            self.conv_dims = [self.embedding_out_dim, self.embedding_out_dim]

        else:
            self.item_embedding_layer = tf.keras.layers.Embedding(
                self.item_num + 1,
                self.embedding_dim,
                name="item_embeddings",
                mask_zero=True,
                embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            )
            target_vocab_dimension = self.item_num + 1
            self.embedding_out_dim = self.embedding_dim

        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.encoder = Encoder(
            self.num_blocks,
            self.seq_max_len,
            self.embedding_out_dim,
            self.hidden_dim,
            self.rnn_name,
            self.conv_dims,
            self.dropout_rate,
        )
        self.mask_layer = tf.keras.layers.Masking(mask_value=0)
        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.embedding_out_dim, 1e-08
        )
        self.linear = tf.keras.layers.Dense(self.tgt_seq_len, activation="relu")
        if self.predict_proba:
            self.final = tf.keras.layers.Dense(
                target_vocab_dimension, activation="softmax"
            )
        else:
            self.final = tf.keras.layers.Dense(
                target_vocab_dimension, activation="linear"
            )

    def process_session_multi(self, input_seq, training):
        mask = tf.expand_dims(
            tf.cast(tf.not_equal(input_seq[:, :, 0], 0), tf.float32), -1
        )

        seq_embeddings = []
        for ii in range(len(self.item_num)):
            emb_ii = self.item_embedding_layer[ii](input_seq[:, :, ii])
            emb_ii = emb_ii * (self.embedding_dims[ii] ** 0.5)
            seq_embeddings.append(emb_ii)

        seq_embeddings = tf.concat(seq_embeddings, axis=-1)
        seq_embeddings = self.dropout_layer(seq_embeddings)

        seq_embeddings *= mask

        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)
        return seq_attention

    def process_session(self, input_seq, training):
        if self.multi_feature:
            return self.process_session_multi(input_seq, training)

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings = self.item_embedding_layer(input_seq)
        seq_embeddings = seq_embeddings * (self.embedding_dim ** 0.5)
        seq_embeddings = self.dropout_layer(seq_embeddings)
        seq_embeddings *= mask

        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)
        return seq_attention

    def call(self, x, training):

        seq_attention = self.process_session(x, training)
        seq_attention = tf.transpose(seq_attention, perm=[0, 2, 1])  # (b, d, s)
        seq_attention = self.linear(seq_attention)  # (b, d, s2)
        seq_attention = tf.transpose(seq_attention, perm=[0, 2, 1])  # (b, s2, d)

        output = self.final(seq_attention)  # (b, s2, |V|)

        return output

    def predict(self, x):
        training = False
        seq_attention = self.process_session(x, training)  # (b, s, d)
        seq_attention = tf.transpose(seq_attention, perm=[0, 2, 1])  # (b, d, s)
        seq_attention = self.linear(seq_attention)  # (b, d, s2)
        seq_attention = tf.transpose(seq_attention, perm=[0, 2, 1])  # (b, s2, d)

        output = self.final(seq_attention)  # (b, s2, |V|)

        return output
