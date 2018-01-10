# -----------------------------------------------------------------------------
#   @brief:
#       For the baseline network, we only need three functions to be defined
#       @fit, @save_checkpoint, and @load_checkpoint
#   @author:
#       Tingwu Wang, modified from kvfran and ppo repository.
# -----------------------------------------------------------------------------
import init_path
import tensorflow as tf
from util import logger
from policy_network import policy_network


class tf_baseline_network(policy_network):
    '''
        @brief:
            Note that the structure of the network is exactly the same as the
            policy_network. So we re-use the structure
    '''

    def __init__(self,
                 session,
                 name_scope,
                 input_size,
                 ob_placeholder=None,
                 trainable=True,
                 args=None):

        self._base_dir = init_path.get_abs_base_dir()
        self._use_ppo = True
        self._ppo_clip = args.ppo_clip
        self._output_size = 1

        super(tf_baseline_network, self).__init__(
            session=session,
            name_scope=name_scope,
            input_size=input_size,
            output_size=self._output_size,
            ob_placeholder=ob_placeholder,
            trainable=trainable,
            build_network_now=True,
            define_std=False,
            is_baseline=True,
            args=args
        )

        self._build_train_placeholders()

        self._build_loss()

    def _build_train_placeholders(self):
        # the names are defined in policy network, reshape to [None]
        self._vpred = tf.reshape(self._action_mu_output, [-1])

        self._target_returns = \
            tf.placeholder(tf.float32, shape=[None], name="target_returns")

    def _build_loss(self):
        '''
            @brief: note that the value clip idea is also used here!
        '''
        # if not self._use_ppo:
        self._loss = tf.reduce_mean(
            tf.square(self._vpred - self._target_returns)
        )

    def predict(self, path):
        # prepare the obs into shape [Batch_size, ob_size] float32 variables
        obs = path['obs'].astype('float32')
        obs = obs.reshape(obs.shape[0], -1)
        return self._session.run(self._vpred, feed_dict={self._input: obs})

    def get_vf_loss(self):
        return self._loss

    def get_target_return_placeholder(self):
        return self._target_returns

    def get_ob_placeholder(self):
        return self._input

    def get_vpred_placeholder(self):
        return self._vpred

    def fit(self):
        '''
            @brief:
                It is a dummy function for the compatibility of the program.
                As for the numpy baseline function, we run @fit to update.
        '''
        logger.warning('This is a dummy function!')
