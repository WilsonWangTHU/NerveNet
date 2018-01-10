# ------------------------------------------------------------------------------
#   @author:
#       Tingwu Wang, modified from the code of Renjie Liao.
#   @UPDATE:
#       27th, Aug: MAJOR UPDATE: change the node input totally.
#       NOTE: Now the input is in node order
# ------------------------------------------------------------------------------
import init_path
import tensorflow as tf
import numpy as np
from policy_network import policy_network
from util import logger
from graph_util import mujoco_parser
from graph_util import gnn_util
from util import nn_cells as nn
from six.moves import xrange


class GGNN(policy_network):
    '''
        @brief:
            Gated Graph Sequence Neural Networks.
            Li, Y., Tarlow, D., Brockschmidt, M. and Zemel, R., 2015.
            arXiv preprint arXiv:1511.05493.
    '''

    def __init__(self,
                 session,
                 name_scope,
                 input_size,
                 output_size,
                 weight_init_methods='orthogonal',
                 ob_placeholder=None,
                 trainable=True,
                 build_network_now=True,
                 is_baseline=False,
                 placeholder_list=None,
                 args=None
                 ):
        '''
            @input: the same as the ones defined in "policy_network"
        '''
        self._shared_network = args.shared_network
        self._node_update_method = args.node_update_method

        policy_network.__init__(
            self,
            session,
            name_scope,
            input_size,
            output_size,
            ob_placeholder=ob_placeholder,
            trainable=trainable,
            build_network_now=False,
            define_std=True,
            is_baseline=False,
            args=args
        )

        self._base_dir = init_path.get_abs_base_dir()
        self._root_connection_option = args.root_connection_option
        self._num_prop_steps = args.gnn_num_prop_steps
        self._init_method = weight_init_methods
        self._gnn_node_option = args.gnn_node_option
        self._gnn_output_option = args.gnn_output_option
        self._gnn_embedding_option = args.gnn_embedding_option
        self._is_baseline = is_baseline
        self._placeholder_list = placeholder_list

        # parse the network shape and do validation check
        self._network_shape = args.network_shape
        self._hidden_dim = args.gnn_node_hidden_dim
        self._input_feat_dim = args.gnn_input_feat_dim
        self._input_obs = ob_placeholder
        self._seed = args.seed
        self._npr = np.random.RandomState(args.seed)

        assert self._input_feat_dim == self._hidden_dim
        logger.info('Network shape is {}'.format(self._network_shape))

        if build_network_now:
            self._build_model()
            if self._shared_network:
                # build the baseline loss and placeholder
                self._build_baseline_train_placeholders()
                self._build_baseline_loss()

    def get_gnn_idx_placeholder(self):
        '''
            @brief: return the placeholders to the agent to construct feed dict
        '''
        return self._receive_idx, self._send_idx, \
            self._node_type_idx, self._inverse_node_type_idx, \
            self._output_type_idx, self._inverse_output_type_idx, \
            self._batch_size_int

    def get_node_info(self):
        return self._node_info

    def _build_model(self):
        '''
            @brief: everything about the network goes here
        '''
        with tf.get_default_graph().as_default():
            tf.set_random_seed(self._seed)

            # record the iteration count
            self._iteration = tf.Variable(0, trainable=False, name='step')

            # read from the xml files
            self._parse_mujoco_template()

            # prepare the network's input and output
            self._prepare()

            # define the network here
            self._build_network_weights()
            self._build_network_graph()

            # get the variable list ready
            self._set_var_list()

    def get_input_obs_placeholder(self):
        return self._input_obs

    def get_input_parameters_placeholder(self):
        return self._input_parameters

    def _build_baseline_loss(self):
        self._baseline_loss = tf.reduce_mean(
            tf.square(self._vpred - self._target_returns)
        )

    def _build_baseline_train_placeholders(self):
        self._target_returns = tf.placeholder(tf.float32, shape=[None],
                                              name='target_returns')

    def get_vf_loss(self):
        return self._baseline_loss

    def get_target_return_placeholder(self):
        return self._target_returns

    def get_ob_placeholder(self):
        return self._input

    def get_vpred_placeholder(self):
        return self._vpred

    def predict(self, feed_dict):
        '''
            @brief:
                generate the baseline function. This is only usable for baseline
                function
        '''
        baseline = self._session.run(self._vpred, feed_dict=feed_dict)
        baseline = baseline.reshape([-1])
        return baseline

    def _prepare(self):
        '''
            @brief:
                get the input placeholders ready. The _input placeholder has
                different size from the input we use for the general network.
        '''
        if self._is_baseline:
            self._receive_idx, self._send_idx, \
                self._node_type_idx, self._inverse_node_type_idx, \
                self._output_type_idx, self._inverse_output_type_idx, \
                self._batch_size_int, \
                self._input_obs, self._input_parameters = self._placeholder_list
        else:
            # step 1: build the input_obs and input_parameters
            if self._input_obs is None:
                self._input_obs = {
                    node_type: tf.placeholder(
                        tf.float32,
                        [None, self._node_info['ob_size_dict'][node_type]],
                        name='input_ob_placeholder_ggnn'
                    )
                    for node_type in self._node_info['node_type_dict']
                }
            else:
                assert False, logger.error('Input mustnt be given to the ggnn')

            input_parameter_dtype = tf.int32 \
                if 'noninput' in self._gnn_embedding_option else tf.float32
            self._input_parameters = {
                node_type: tf.placeholder(
                    input_parameter_dtype,
                    [None, self._node_info['para_size_dict'][node_type]],
                    name='input_para_placeholder_ggnn')
                for node_type in self._node_info['node_type_dict']
            }

            # step 2: the receive and send index
            self._receive_idx = tf.placeholder(
                tf.int32, shape=(None), name='receive_idx'
            )
            self._send_idx = {
                edge_type: tf.placeholder(
                    tf.int32, shape=(None),
                    name='send_idx_{}'.format(edge_type))
                for edge_type in self._node_info['edge_type_list']
            }

            # step 3: the node type index and inverse node type index
            self._node_type_idx = {
                node_type: tf.placeholder(
                    tf.int32, shape=(None),
                    name='node_type_idx_{}'.format(node_type))
                for node_type in self._node_info['node_type_dict']
            }
            self._inverse_node_type_idx = tf.placeholder(
                tf.int32, shape=(None), name='inverse_node_type_idx'
            )

            # step 4: the output node index
            self._output_type_idx = {
                output_type: tf.placeholder(
                    tf.int32, shape=(None),
                    name='output_type_idx_{}'.format(output_type)
                )
                for output_type in self._node_info['output_type_dict']
            }

            self._inverse_output_type_idx = tf.placeholder(
                tf.int32, shape=(None), name='inverse_output_type_idx'
            )

            # step 5: batch_size
            self._batch_size_int = tf.placeholder(
                tf.int32, shape=(), name='batch_size_int'
            )

    def _build_network_weights(self):
        '''
            @brief: build the network
        '''
        # step -1: build the baseline network mlp if needed
        if self._shared_network:
            MLP_baseline_shape = self._network_shape + [1] + \
                [self._hidden_dim]  # (l_1, l_2, ..., l_o, l_i)
            MLP_baseline_act_func = ['tanh'] * (len(MLP_baseline_shape) - 1)
            MLP_baseline_act_func[-1] = None
            with tf.variable_scope('baseline'):
                self._MLP_baseline_out = nn.MLP(
                    MLP_baseline_shape, init_method=self._init_method,
                    act_func=MLP_baseline_act_func, add_bias=True, scope='vpred'
                )

        # step 1: build the weight parameters (mlp, gru)
        with tf.variable_scope(self._name_scope):
            # step 1_1: build the embedding matrix (mlp)
            # tensor shape (None, para_size) --> (None, input_dim - ob_size)
            assert self._input_feat_dim % 2 == 0
            if 'noninput' not in self._gnn_embedding_option:
                self._MLP_embedding = {
                    node_type: nn.MLP(
                        [self._input_feat_dim / 2,
                         self._node_info['para_size_dict'][node_type]],
                        init_method=self._init_method,
                        act_func=['tanh'] * 1,  # one layer at most
                        add_bias=True,
                        scope='MLP_embedding_node_type_{}'.format(node_type)
                    )
                    for node_type in self._node_info['node_type_dict']
                    if self._node_info['ob_size_dict'][node_type] > 0
                }
                self._MLP_embedding.update({
                    node_type: nn.MLP(
                        [self._input_feat_dim,
                         self._node_info['para_size_dict'][node_type]],
                        init_method=self._init_method,
                        act_func=['tanh'] * 1,  # one layer at most
                        add_bias=True,
                        scope='MLP_embedding_node_type_{}'.format(node_type)
                    )
                    for node_type in self._node_info['node_type_dict']
                    if self._node_info['ob_size_dict'][node_type] == 0
                })
            else:
                embedding_vec_size = max(
                    np.reshape(
                        [max(self._node_info['node_parameters'][i_key])
                         for i_key in self._node_info['node_parameters']],
                        [-1]
                    )
                ) + 1
                embedding_vec_size = int(embedding_vec_size)
                self._embedding_variable = {}
                out = self._npr.randn(
                    embedding_vec_size, self._input_feat_dim / 2
                ).astype(np.float32)
                out *= 1.0 / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
                self._embedding_variable[False] = tf.Variable(
                    out, name='embedding_HALF', trainable=self._trainable
                )

                if np.any([node_size == 0 for _, node_size
                           in self._node_info['ob_size_dict'].iteritems()]):

                    out = self._npr.randn(
                        embedding_vec_size, self._input_feat_dim
                    ).astype(np.float32)
                    out *= 1.0 / np.sqrt(np.square(out).sum(axis=0,
                                                            keepdims=True))
                    self._embedding_variable[True] = tf.Variable(
                        out, name='embedding_FULL', trainable=self._trainable
                    )

            # step 1_2: build the ob mapping matrix (mlp)
            # tensor shape (None, para_size) --> (None, input_dim - ob_size)
            self._MLP_ob_mapping = {
                node_type: nn.MLP(
                    [self._input_feat_dim / 2,
                     self._node_info['ob_size_dict'][node_type]],
                    init_method=self._init_method,
                    act_func=['tanh'] * 1,  # one layer at most
                    add_bias=True,
                    scope='MLP_embedding_node_type_{}'.format(node_type)
                )
                for node_type in self._node_info['node_type_dict']
                if self._node_info['ob_size_dict'][node_type] > 0
            }

        with tf.variable_scope(self._name_scope):
            # step 1_4: build the mlp for the propogation between nodes
            MLP_prop_shape = self._network_shape + \
                [self._hidden_dim] + [self._hidden_dim]
            self._MLP_prop = {
                i_edge: nn.MLP(
                    MLP_prop_shape,
                    init_method=self._init_method,
                    act_func=['tanh'] * (len(MLP_prop_shape) - 1),
                    add_bias=True,
                    scope='MLP_prop_edge_{}'.format(i_edge)
                )
                for i_edge in self._node_info['edge_type_list']
            }

            logger.info('building prop mlp for edge type {}'.format(
                self._node_info['edge_type_list'])
            )

            # step 1_5: build the node update function for each node type
            if self._node_update_method == 'GRU':
                self._Node_update = {
                    i_node_type: nn.GRU(
                        self._hidden_dim,
                        self._hidden_dim,
                        init_method=self._init_method,
                        scope='GRU_node_{}'.format(i_node_type)
                    )
                    for i_node_type in self._node_info['node_type_dict']
                }
            else:
                assert self._node_update_method == 'MLP'
                hidden_MLP_update_shape = self._network_shape
                self._Node_update = {
                    i_node_type: nn.MLPU(
                        message_dim=self._hidden_dim,
                        embedding_dim=self._hidden_dim,
                        hidden_shape=hidden_MLP_update_shape,
                        init_method=self._init_method,
                        act_func_type='tanh',
                        add_bias=True,
                        scope='MLPU_node_{}'.format(i_node_type)
                    )
                    for i_node_type in self._node_info['node_type_dict']
                }

            logger.info('building node update function for node type {}'.format(
                self._node_info['node_type_dict'])
            )

            # step 1_6: the mlp for the mu of the actions
            MLP_out_shape = self._network_shape + [1] + \
                [self._hidden_dim]  # (l_1, l_2, ..., l_o, l_i)
            MLP_out_act_func = ['tanh'] * (len(MLP_out_shape) - 1)
            MLP_out_act_func[-1] = None

            if not self._is_baseline:
                self._MLP_Out = {
                    output_type: nn.MLP(
                        MLP_out_shape,
                        init_method=self._init_method,
                        act_func=MLP_out_act_func,
                        add_bias=True,
                        scope='MLP_out'
                    )
                    for output_type in self._node_info['output_type_dict']
                }

                # step 1_7 (optional): the mlp for the log std of the actions
            else:
                self._MLP_Out = nn.MLP(
                    MLP_out_shape,
                    init_method=self._init_method,
                    act_func=MLP_out_act_func,
                    add_bias=True,
                    scope='MLP_out'
                )

            # step 1_8: build the log std for the actions
            with tf.variable_scope(self._name_scope):
                # size: [1, num_action]
                self._action_dist_logstd = tf.Variable(
                    (0.0 * self._npr.randn(1, self._output_size)).astype(
                        np.float32
                    ),
                    name="policy_logstd",
                    trainable=self._trainable
                )

    def _build_network_graph(self):
        # step 2: gather the input_feature from obs and node parameters
        if 'noninput' not in self._gnn_embedding_option:
            self._input_embedding = {
                node_type: self._MLP_embedding[node_type](
                    self._input_parameters[node_type]
                )[-1]
                for node_type in self._node_info['node_type_dict']
            }
        else:
            self._input_embedding = {
                node_type: tf.gather(
                    self._embedding_variable[
                        self._node_info['ob_size_dict'][node_type] == 0
                    ],
                    tf.reshape(self._input_parameters[node_type], [-1])
                )
                for node_type in self._node_info['node_type_dict']
            }

        self._ob_feat = {
            node_type: self._MLP_ob_mapping[node_type](
                self._input_obs[node_type]
            )[-1]
            for node_type in self._node_info['node_type_dict']
            if self._node_info['ob_size_dict'][node_type] > 0
        }
        self._ob_feat.update({
            node_type: self._input_obs[node_type]
            for node_type in self._node_info['node_type_dict']
            if self._node_info['ob_size_dict'][node_type] == 0
        })

        self._input_feat = {  # shape: [node_num, embedding_size + ob_size]
            node_type: tf.concat(
                [self._input_embedding[node_type], self._ob_feat[node_type]],
                axis=1
            )
            for node_type in self._node_info['node_type_dict']
        }
        self._input_node_hidden = self._input_feat

        self._input_node_hidden = tf.concat(
            [self._input_node_hidden[node_type]
             for node_type in self._node_info['node_type_dict']],
            axis=0
        )
        self._input_node_hidden = tf.gather(  # get node order into graph order
            self._input_node_hidden,
            self._inverse_node_type_idx,
            name='get_order_back_gather_init'
        )

        # step 3: unroll the propogation
        self._node_hidden = [None] * (self._num_prop_steps + 1)
        self._node_hidden[-1] = self._input_node_hidden  # trick to use [-1]
        self._prop_msg = [None] * self._node_info['num_edge_type']

        for tt in xrange(self._num_prop_steps):
            ee = 0
            # TODO: change to enumerate
            for i_edge_type in self._node_info['edge_type_list']:
                node_activate = \
                    tf.gather(
                        self._node_hidden[tt - 1],
                        self._send_idx[i_edge_type],
                        name='edge_id_{}_prop_steps_{}'.format(i_edge_type, tt)
                    )
                self._prop_msg[ee] = \
                    self._MLP_prop[i_edge_type](node_activate)[-1]

                ee += 1

            # aggregate messages
            concat_msg = tf.concat(self._prop_msg, 0)
            self.concat_msg = concat_msg
            message = tf.unsorted_segment_sum(
                concat_msg, self._receive_idx,
                self._node_info['num_nodes'] * self._batch_size_int
            )
            denom_const = tf.unsorted_segment_sum(
                tf.ones_like(concat_msg), self._receive_idx,
                self._node_info['num_nodes'] * self._batch_size_int
            )
            message = tf.div(message, (denom_const + tf.constant(1.0e-10)))

            # update the hidden states via GRU
            new_state = []
            for i_node_type in self._node_info['node_type_dict']:
                new_state.append(
                    self._Node_update[i_node_type](
                        tf.gather(
                            message,
                            self._node_type_idx[i_node_type],
                            name='GRU_message_node_type_{}_prop_step_{}'.format(
                                i_node_type, tt
                            )
                        ),
                        tf.gather(
                            self._node_hidden[tt - 1],
                            self._node_type_idx[i_node_type],
                            name='GRU_feat_node_type_{}_prop_steps_{}'.format(
                                i_node_type, tt
                            )
                        )
                    )
                )
            new_state = tf.concat(new_state, 0)  # BTW, the order is wrong
            # now, get the orders back
            self._node_hidden[tt] = tf.gather(
                new_state, self._inverse_node_type_idx,
                name='get_order_back_gather_prop_steps_{}'.format(tt)
            )

        # step 3: get the output
        self._final_node_hidden = self._node_hidden[-2]
        if self._is_baseline:
            self._final_node_hidden = tf.reshape(
                self._final_node_hidden,
                [self._batch_size_int, self._node_info['num_nodes'], -1]
            )
            self.final_root_hidden = tf.reshape(
                self._final_node_hidden[:, 0, :],
                [self._batch_size_int, -1]
            )
            self._action_mu_output = self._MLP_Out(self.final_root_hidden)[-1]
        else:

            self._action_mu_output = []
            for output_type in self._node_info['output_type_dict']:
                self._action_mu_output.append(
                    self._MLP_Out[output_type](
                        tf.gather(
                            self._final_node_hidden,
                            self._output_type_idx[output_type],
                            name='output_type_{}'.format(output_type)
                        )
                    )[-1]
                )

            self._action_mu_output = tf.concat(self._action_mu_output, 0)
            self._action_mu_output = tf.gather(
                self._action_mu_output,
                self._inverse_output_type_idx,
                name='output_inverse'
            )

            self._action_mu_output = tf.reshape(self._action_mu_output,
                                                [self._batch_size_int, -1])

            # step 4: build the log std for the actions
            self._action_dist_logstd_param = tf.tile(
                self._action_dist_logstd,
                tf.stack((tf.shape(self._action_mu_output)[0], 1))
            )

        if self._shared_network:
            self._final_node_hidden = tf.reshape(
                self._final_node_hidden,
                [self._batch_size_int, self._node_info['num_nodes'], -1]
            )
            self.final_root_hidden = tf.reshape(
                self._final_node_hidden[:, 0, :],
                [self._batch_size_int, -1]
            )
            self._vpred = self._MLP_baseline_out(self.final_root_hidden)[-1]

    def _parse_mujoco_template(self):
        '''
            @brief:
                In this function, we construct the dict for node information.
                The structure is _node_info
            @attribute:
                1. general informatin about the graph
                    @self._node_info['tree']
                    @self._node_info['debug_info']
                    @self._node_info['relation_matrix']

                2. information about input output
                    @self._node_info['input_dict']:
                        self._node_info['input_dict'][id_of_node] is a list of
                        ob positions
                    @self._node_info['output_list']

                3. information about the node
                    @self._node_info['node_type_dict']:
                        self._node_info['node_type_dict']['body'] is a list of
                        node id
                    @self._node_info['num_nodes']

                4. information about the edge
                    @self._node_info['edge_type_list'] = self._edge_type_list
                        the list of edge ids
                    @self._node_info['num_edges']
                    @self._node_info['num_edge_type']

                6. information about the index
                    @self._node_info['node_in_graph_list']
                        The order of nodes if placed by types ('joint', 'body')
                    @self._node_info['inverse_node_list']
                        The inverse of 'node_in_graph_list'
                    @self._node_info['receive_idx'] = receive_idx
                    @self._node_info['receive_idx_raw'] = receive_idx_raw
                    @self._node_info['send_idx'] = send_idx

                7. information about the embedding size and ob size
                    @self._node_info['para_size_dict']
                    @self._node_info['ob_size_dict']
            '''
        # step 0: parse the mujoco xml
        self._node_info = mujoco_parser.parse_mujoco_graph(
            self._task_name,
            gnn_node_option=self._gnn_node_option,
            root_connection_option=self._root_connection_option,
            gnn_output_option=self._gnn_output_option,
            gnn_embedding_option=self._gnn_embedding_option
        )

        # step 1: check that the input and output size is matched
        gnn_util.io_size_check(self._input_size, self._output_size,
                               self._node_info, self._is_baseline)

        # step 2: check for ob size for each node type, construct the node dict
        self._node_info = gnn_util.construct_ob_size_dict(self._node_info,
                                                          self._input_feat_dim)

        # step 3: get the inverse node offsets (used to construct gather idx)
        self._node_info = gnn_util.get_inverse_type_offset(self._node_info,
                                                           'node')

        # step 4: get the inverse node offsets (used to gather output idx)
        self._node_info = gnn_util.get_inverse_type_offset(self._node_info,
                                                           'output')

        # step 5: register existing edge and get the receive and send index
        self._node_info = gnn_util.get_receive_send_idx(self._node_info)

    def get_num_nodes(self):
        return self._node_info['num_nodes']

    def get_logstd(self):
        return self._action_dist_logstd

    def set_logstd(self, value):
        var_shape = (self._action_dist_logstd.get_shape()).as_list()
        new_value = np.ones(var_shape) * value
        self._session.run(self._action_dist_logstd.assign(new_value))

    def _set_var_list(self):
        # collect the tf variable and the trainable tf variable
        self._trainable_var_list = [var for var in tf.trainable_variables()
                                    if self._name_scope in var.name]

        self._all_var_list = [var for var in tf.global_variables()
                              if self._name_scope in var.name]
