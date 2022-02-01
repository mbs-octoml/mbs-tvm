import argparse
import tensorflow as tf
import numpy as np
import time
from shared_functions import make_matmul, measure_tf2_gpu
import os, math

this_code_path = os.path.dirname(os.path.abspath(__file__))
#model_path = f"{this_code_path}/../onnx/bert_full.pb"
#model = tf.saved_model.load(model_path)

import torch

def make_linear(input_tensor, out_channels):
    weight_shape = (input_tensor.shape[0], input_tensor.shape[2], out_channels)
    bias_shape = (input_tensor.shape[0], 1, out_channels)
    weight = tf.constant(np.random.random_sample(weight_shape), dtype=tf.float32)
    bias = tf.constant(np.random.random_sample(bias_shape), dtype=tf.float32)
    return tf.math.add(tf.matmul(input_tensor, weight), bias)
 

class LayerNorm(object):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = tf.constant(torch.ones(features).numpy())
        self.b_2 = tf.constant(torch.zeros(features).numpy())
        self.eps = eps

    def forward(self, x):
        mean, var = tf.nn.moments(x, len(x.shape) - 1, shift=None, keepdims=True, name=None)
        std = tf.sqrt(var)
        #mean = x.mean(-1, keepdim=True)
        #std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GELU(object):
    def forward(self, x):
        return 0.5 * x * (1 + tf.math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.math.pow(x, 3))))

class PositionwiseFeedForward(object):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = lambda x: make_linear(x, d_ff)
        self.w_2 = lambda x: make_linear(x, d_model)
        # self.dropout = nn.Dropout(dropout)
        # self.activation = GELU()

    def forward(self, x):
        # return self.w_2(self.dropout(self.activation(self.w_1(x))))
        return self.w_2(tf.nn.relu(self.w_1(x)))

class SublayerConnection(object):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # return x + self.dropout(sublayer(self.norm(x)))
        return x + sublayer(self.norm.forward(x))
        # return x + sublayer(x)

class SegmentEmbedding(object):
    def __init__(self, embed_size=512):
        self.embeds = tf.constant(np.random.random_sample((3,embed_size)), dtype=tf.float32)
        #super().__init__(3, embed_size, padding_idx=0)
    def forward(self, x):
        return tf.nn.embedding_lookup(
            self.embeds, x, max_norm=None, name=None
        )


class TokenEmbedding(object):
    def __init__(self, vocab_size, embed_size=512):
        #super().__init__(vocab_size, embed_size, padding_idx=0)
        self.embeds = tf.constant(np.random.random_sample((vocab_size,embed_size)), dtype=tf.float32)
    def forward(self, x):
        return tf.nn.embedding_lookup(
            self.embeds, x, max_norm=None, name=None
        )    

class PositionalEmbedding(object):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.pe = tf.constant(self.pe.numpy())

    def forward(self, x):
        return self.pe[:, :x.shape[1]]

class BERTEmbedding(object):
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        # self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token.forward(sequence) + self.position.forward(sequence) + self.segment.forward(segment_label)
        return x
        # return self.dropout(x)

class Attention(object):
    def forward(self, query, key, value, mask=None, dropout=None):
        ks = key.shape
        scores = tf.matmul(query, tf.transpose(key, perm=[0,1,3,2]) ) \
                 / math.sqrt(query.shape[-1])

        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)
        #print(scores.shape)

        p_attn = tf.nn.softmax(scores, axis=3)

        # if dropout is not None:
        #     p_attn = dropout(p_attn)

        return tf.matmul(p_attn, value), p_attn

class MultiHeadedAttention(object):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = [lambda x: make_linear(x, d_model) for _ in range(3)]
        self.output_linear = lambda x: make_linear(x, d_model)
        self.attention = Attention()

        #self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # print(batch_size)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [tf.transpose(tf.reshape(l(x),(batch_size, -1, self.h, self.d_k)), perm=[0,2,1,3])
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention.forward(query, key, value, mask=mask, dropout=0)

        # 3) "Concat" using a view and apply a final linear.
        #x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = tf.reshape(tf.transpose(x,perm=[0,2,1,3]),(batch_size, -1, self.h * self.d_k))
        return self.output_linear(x)

class TransformerBlock(object):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, mask):
        x = self.input_sublayer.forward(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer.forward(x, self.feed_forward.forward)
        return x
        # return self.dropout(x)


class BERTFULL(object):
    def __init__(self, hidden=256, n_layers=8, attn_heads=8, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        # self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)]

    # def forward(self, x, segment_info):
    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask = None

        # embedding the indexed sequence to sequence of vectors
        # x = self.embedding(x, segment_info)
        # print("x: ", x.shape)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

model = BERTFULL()

def bert_full_tf2_model(input):
    return model.forward(input)

# @tf.function(jit_compile=False)
@tf.function(experimental_compile=False)
def bert_full_tf2(input):
    return bert_full_tf2_model(input)

# @tf.function(jit_compile=True)
@tf.function(experimental_compile=True)
def bert_full_tf2_xla(input):
    return bert_full_tf2_model(input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    args.network = 'bert_full'
    input_shape = (1, 64, 256)
    inputs = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    method_name = 'TF'
    measure_tf2_gpu(bert_full_tf2, inputs, method_name, args)

    method_name = 'TF-XLA'
    measure_tf2_gpu(bert_full_tf2_xla, inputs, method_name, args)
