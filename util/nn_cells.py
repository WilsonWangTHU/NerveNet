"""
Basic cells of neural network

Renjie Liao
"""
import tensorflow as tf
from six.moves import xrange


def weight_variable(shape,
                    init_method=None,
                    dtype=tf.float32,
                    init_para=None,
                    wd=None,
                    seed=1234,
                    name=None,
                    trainable=True):
    """ Initialize Weights

        Input:
                shape: list of int, shape of the weights
                init_method: string, indicates initialization method
                init_para: a dictionary,
                init_val: if it is not None, it should be a tensor
                wd: a float, weight decay
                name:
                trainable:

        Output:
                var: a TensorFlow Variable
    """

    if init_method is None:
        initializer = tf.zeros_initializer(shape, dtype=dtype)
    elif init_method == "normal":
        initializer = tf.random_normal_initializer(
            mean=init_para["mean"],
            stddev=init_para["stddev"],
            seed=seed,
            dtype=dtype
        )
    elif init_method == "truncated_normal":
        initializer = tf.truncated_normal_initializer(
            mean=init_para["mean"],
            stddev=init_para["stddev"],
            seed=seed,
            dtype=dtype
        )
    elif init_method == "uniform":
        initializer = tf.random_uniform_initializer(
            minval=init_para["minval"],
            maxval=init_para["maxval"],
            seed=seed,
            dtype=dtype
        )
    elif init_method == "constant":
        initializer = tf.constant_initializer(value=init_para["val"],
                                              dtype=dtype)
    elif init_method == "xavier":
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=True, seed=seed, dtype=dtype
        )
    elif init_method == 'orthogonal':
        initializer = tf.orthogonal_initializer(
            gain=1.0, seed=seed, dtype=dtype
        )
    else:
        raise ValueError("Unsupported initialization method!")

    var = tf.Variable(initializer(shape), name=name, trainable=trainable)

    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name="weight_decay")
        tf.add_to_collection("losses", weight_decay)

    return var


class GRU(object):
    """ Gated Recurrent Units (GRU)

        Input:
                input_dim: input dimension
                hidden_dim: hidden dimension
                wd: a float, weight decay
                scope: tf scope of the model

        Output:
                a function which computes the output of GRU with one step
    """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 init_method,
                 wd=None,
                 dtype=tf.float32,
                 init_std=None,
                 scope="GRU"):

        self._init_method = init_method

        # initialize variables
        with tf.variable_scope(scope):
            self._w_xi = weight_variable(
                [input_dim, hidden_dim], init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd, name="w_xi", dtype=dtype
            )
            self._w_hi = weight_variable(
                [hidden_dim, hidden_dim], init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd, name="w_hi", dtype=dtype
            )
            self._b_i = weight_variable(
                [hidden_dim], init_method="constant",
                init_para={"val": 0.0}, wd=wd, name="b_i", dtype=dtype
            )

            self._w_xr = weight_variable(
                [input_dim, hidden_dim], init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd, name="w_xr", dtype=dtype
            )
            self._w_hr = weight_variable(
                [hidden_dim, hidden_dim], init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd, name="w_hr", dtype=dtype
            )
            self._b_r = weight_variable(
                [hidden_dim], init_method="constant", init_para={"val": 0.0},
                wd=wd, name="b_r", dtype=dtype
            )

            self._w_xu = weight_variable(
                [input_dim, hidden_dim], init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd, name="w_xu", dtype=dtype
            )
            self._w_hu = weight_variable(
                [hidden_dim, hidden_dim], init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd, name="w_hu", dtype=dtype
            )
            self._b_u = weight_variable(
                [hidden_dim], init_method="constant", init_para={"val": 0.0},
                wd=wd, name="b_u", dtype=dtype
            )

    def __call__(self, x, state, is_batch_mul=False):
        # update gate
        g_i = tf.sigmoid(
            tf.matmul(x, self._w_xi) + tf.matmul(state, self._w_hi) + self._b_i)

        # reset gate
        g_r = tf.sigmoid(
            tf.matmul(x, self._w_xr) + tf.matmul(state, self._w_hr) + self._b_r)

        # new memory implementation 1
        u = tf.tanh(
            tf.matmul(x, self._w_xu) + tf.matmul(g_r * state, self._w_hu) +
            self._b_u)

        # hidden state
        new_state = state * g_i + u * (1 - g_i)

        return new_state


class MLP(object):
    """ Multi Layer Perceptron (MLP)
                Note: the number of layers is N

        Input:
                dims: a list of N+1 int, number of hidden units (last one is the
                input dimension)
                act_func: a list of N activation functions
                add_bias: a boolean, indicates whether adding bias or not
                wd: a float, weight decay
                scope: tf scope of the model

        Output:
                a function which outputs a list of N tensors, each is the hidden
                activation of one layer
    """

    def __init__(self,
                 dims,
                 init_method,
                 act_func=None,
                 add_bias=True,
                 wd=None,
                 dtype=tf.float32,
                 init_std=None,
                 scope="MLP"):

        self._init_method = init_method

        self._scope = scope
        self._add_bias = add_bias
        self._num_layer = len(dims) - 1  # the last one is the input dim
        self._w = [None] * self._num_layer
        self._b = [None] * self._num_layer
        self._act_func = [None] * self._num_layer

        # initialize variables
        with tf.variable_scope(scope):
            for ii in xrange(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    dim_in = dims[ii - 1]
                    dim_out = dims[ii]

                    self._w[ii] = weight_variable(
                        [dim_in, dim_out], init_method=self._init_method,
                        init_para={"mean": 0.0, "stddev": init_std},
                        wd=wd, name="w", dtype=dtype
                    )

                    if add_bias:
                        self._b[ii] = weight_variable(
                            [dim_out], init_method="constant",
                            init_para={"val": 1.0e-2},
                            wd=wd, name="b", dtype=dtype
                        )

                    if act_func and act_func[ii] is not None:
                        if act_func[ii] == "relu":
                            self._act_func[ii] = tf.nn.relu
                        elif act_func[ii] == "sigmoid":
                            self._act_func[ii] = tf.sigmoid
                        elif act_func[ii] == "tanh":
                            self._act_func[ii] = tf.tanh
                        else:
                            raise ValueError("Unsupported activation method!")

    def __call__(self, x):
        h = [None] * self._num_layer

        with tf.variable_scope(self._scope):
            for ii in xrange(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    if ii == 0:
                        input_vec = x
                    else:
                        input_vec = h[ii - 1]

                    h[ii] = tf.matmul(input_vec, self._w[ii])

                    if self._add_bias:
                        h[ii] += self._b[ii]

                    if self._act_func[ii] is not None:
                        h[ii] = self._act_func[ii](h[ii])

        return h


class LSTM(object):
    """ Long Short-term Memory (LSTM)

        Input:
                input_dim: input dimension
                hidden_dim: hidden dimension
                wd: a float, weight decay
                scope: tf scope of the model

        Output:
                a function which computes the output of GRU with one step
    """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 wd=None,
                 dtype=tf.float32,
                 init_std=None,
                 scope="LSTM"):

        if init_std:
            self._init_method = "truncated_normal"
        else:
            self._init_method = "xavier"

        # initialize variables
        with tf.variable_scope(scope):
            # forget gate
            self._Wf = weight_variable(
                [input_dim, hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="Wf",
                dtype=dtype)

            self._Uf = weight_variable(
                [hidden_dim, hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="Uf",
                dtype=dtype)

            self._bf = weight_variable(
                [hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="bf",
                dtype=dtype)

            # input gate
            self._Wi = weight_variable(
                [input_dim, hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="Wi",
                dtype=dtype)

            self._Ui = weight_variable(
                [hidden_dim, hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="Ui",
                dtype=dtype)

            self._bi = weight_variable(
                [hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="bi",
                dtype=dtype)

            # output gate
            self._Wo = weight_variable(
                [input_dim, hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="Wo",
                dtype=dtype)

            self._Uo = weight_variable(
                [hidden_dim, hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="Uo",
                dtype=dtype)

            self._bo = weight_variable(
                [hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="bo",
                dtype=dtype)

            # output gate
            self._Wo = weight_variable(
                [input_dim, hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="Wo",
                dtype=dtype)

            self._Uo = weight_variable(
                [hidden_dim, hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="Uo",
                dtype=dtype)

            self._bo = weight_variable(
                [hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="bo",
                dtype=dtype)

            # cell
            self._Wc = weight_variable(
                [input_dim, hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="Wc",
                dtype=dtype)

            self._Uc = weight_variable(
                [hidden_dim, hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="Uc",
                dtype=dtype)

            self._bc = weight_variable(
                [hidden_dim],
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std},
                wd=wd,
                name="bc",
                dtype=dtype)

    def __call__(self, x, state, memory):
        # forget gate
        f = tf.sigmoid(
            tf.matmul(x, self._Wf) + tf.matmul(state, self._Uf) + self._bf)

        # input gate
        i = tf.sigmoid(
            tf.matmul(x, self._Wi) + tf.matmul(state, self._Ui) + self._bi)

        # output gate
        o = tf.sigmoid(
            tf.matmul(x, self._Wo) + tf.matmul(state, self._Uo) + self._bo)

        # new memory
        new_memory = f * memory + i * tf.tanh(
            tf.matmul(x, self._Wc) + tf.matmul(state, self._Uc) + self._bc)

        # hidden state
        new_state = o * tf.tanh(new_memory)

        return new_state, new_memory


class ResMLP(MLP):

    def __init__(self,
                 dims,
                 act_func=None,
                 add_bias=True,
                 wd=None,
                 dtype=tf.float32,
                 init_std=None,
                 scope="ResMLP"):
        super(ResMLP, self).__init__(
            dims,
            act_func=act_func,
            add_bias=add_bias,
            wd=wd,
            dtype=dtype,
            init_std=init_std,
            scope=scope)

    def __call__(self, x):
        h = [None] * self._num_layer

        with tf.variable_scope(self._scope):
            for ii in xrange(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    if ii == 0:
                        input_vec = x
                    else:
                        input_vec = h[ii - 1]

                    h[ii] = tf.matmul(input_vec, self._w[ii])

                    if self._add_bias:
                        h[ii] += self._b[ii]

                    if self._act_func[ii] is not None:
                        h[ii] = self._act_func[ii](h[ii])

                    # add residual connection
                    if ii > 0 and ii < self._num_layer - 1:
                        h[ii] = h[ii] + h[ii - 1]

        return h


class MLPU(MLP):
    """ MLP update unit

        Input:
                input_dim: input dimension
                hidden_dim: hidden dimension
                wd: a float, weight decay
                scope: tf scope of the model

        Output:
                a function which computes the output of GRU with one step
    """

    def __init__(self,
                 message_dim,
                 embedding_dim,
                 hidden_shape,
                 init_method,
                 act_func_type,
                 add_bias,
                 wd=None,
                 dtype=tf.float32,
                 init_std=None,
                 scope="MLPU"):

        input_dim = message_dim + embedding_dim
        output_dim = embedding_dim
        network_shape = hidden_shape + [output_dim, input_dim]

        MLP.__init__(self,
                     network_shape,
                     init_method,
                     act_func=[act_func_type] * (len(network_shape) - 1),
                     add_bias=True,
                     wd=None,
                     dtype=tf.float32,
                     init_std=None,
                     scope="")

    def __call__(self, message, hidden_embedding):
        mlp_input = tf.concat([message, hidden_embedding], 1)
        return MLP.__call__(self, mlp_input)[-1]
