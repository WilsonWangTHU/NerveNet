# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   TODO: 1. modify the mlp layer setting (list)
# -----------------------------------------------------------------------------
import init_path
import tensorflow as tf
import numpy as np
from util.utils import fully_connected
from util import model_saver


def normc_initializer(npr, std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = npr.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class policy_network(object):
    '''
        @brief:
            In this object class, we define the network structure, the restore
            function and save function.
            It will only be called in the agent/agent.py
    '''

    def __init__(self,
                 session,
                 name_scope,
                 input_size,
                 output_size,
                 ob_placeholder=None,
                 trainable=True,
                 build_network_now=True,
                 define_std=True,
                 is_baseline=False,
                 args=None
                 ):
        '''
            @input:
                @ob_placeholder:
                    if this placeholder is not given, we will make one in this
                    class.

                @trainable:
                    If it is set to true, then the policy weights will be
                    trained. It is useful when the class is a subnet which
                    is not trainable

        '''
        self._session = session
        self._name_scope = name_scope

        self._input_size = input_size
        self._output_size = output_size
        self._base_dir = init_path.get_abs_base_dir()
        self._is_baseline = is_baseline

        self._input = ob_placeholder
        self._trainable = trainable

        self._define_std = define_std

        self._task_name = args.task_name
        self._network_shape = args.network_shape
        self._npr = np.random.RandomState(args.seed)
        self.args = args

        if build_network_now:
            with tf.get_default_graph().as_default():
                tf.set_random_seed(args.seed)
                self._build_model()

    def debug_get_std(self):
        return self._action_dist_logstd_param

    def _build_model(self):
        '''
            @brief: the network is defined here
        '''
        self._iteration = tf.Variable(0, trainable=False, name='step')

        # two initializer
        weight_init = normc_initializer(self._npr)
        bias_init = tf.constant_initializer(0)

        # if input not provided, make one
        if self._input is None:
            self._input = tf.placeholder(tf.float32, [None, self._input_size],
                                         name='ob_input')

        with tf.variable_scope(self._name_scope):
            self._layer = self._input
            self._layer_input_size = self._input_size
            for i_layer in range(len(self._network_shape)):
                self._layer = \
                    fully_connected(self._layer,
                                    self._layer_input_size,
                                    self._network_shape[i_layer],
                                    weight_init,
                                    bias_init, "policy_" + str(i_layer),
                                    trainable=self._trainable)
                self._layer = tf.nn.tanh(self._layer)
                self._layer_input_size = self._network_shape[i_layer]

            # the output layer
            if not self._is_baseline:
                weight_init = normc_initializer(self._npr, 0.01)
            self._action_mu_output = fully_connected(self._layer,
                                                     self._layer_input_size,
                                                     self._output_size,
                                                     weight_init,
                                                     bias_init,
                                                     "policy_output",
                                                     trainable=self._trainable)

            if self._define_std:
                # size: [1, num_action]
                self._action_dist_logstd = tf.Variable(
                    (0 * self._npr.randn(
                        1, self._output_size)).astype(np.float32),
                    name="policy_logstd",
                    trainable=self._trainable
                )

                # size: [batch, num_action]
                self._action_dist_logstd_param = tf.tile(
                    self._action_dist_logstd,
                    tf.stack((tf.shape(self._action_mu_output)[0], 1))
                )

        # get the variable list
        self._set_var_list()

    def _set_var_list(self):
        # collect the tf variable and the trainable tf variable
        self._trainable_var_list = [var for var in tf.trainable_variables()
                                    if self._name_scope in var.name]

        self._all_var_list = [var for var in tf.global_variables()
                              if self._name_scope in var.name]

    def get_action_dist_mu(self):
        return self._action_mu_output

    def get_action_dist_logstd_param(self):
        return self._action_dist_logstd_param

    def get_input_placeholder(self):
        return self._input

    def get_var_list(self):
        return self._trainable_var_list, self._all_var_list

    def get_iteration_var(self):
        return self._iteration

    def load_checkpoint(self, ckpt_path,
                        transfer_env='Nothing2Nothing', logstd_option='load',
                        gnn_option_list=None, mlp_raw_transfer=False):
        # in this function, we check if the checkpoint has been loaded
        model_saver.load_tf_model(self._session, ckpt_path,
                                  tf_var_list=self._all_var_list,
                                  transfer_env=transfer_env,
                                  logstd_option=logstd_option,
                                  gnn_option_list=gnn_option_list,
                                  mlp_raw_transfer=mlp_raw_transfer)

    def save_checkpoint(self, ckpt_path):
        model_saver.save_tf_model(self._session, ckpt_path, self._all_var_list)

    def get_logstd(self):
        return self._action_dist_logstd

    def set_logstd(self, value):
        var_shape = (self._action_dist_logstd.get_shape()).as_list()
        new_value = np.ones(var_shape) * value
        self._session.run(self._action_dist_logstd.assign(new_value))
