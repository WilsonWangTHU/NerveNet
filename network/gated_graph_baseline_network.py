# -----------------------------------------------------------------------------
#   @brief:
#       For the baseline network, we only need three functions to be defined
#       @fit, @save_checkpoint, and @load_checkpoint
#   @author:
#       Tingwu Wang, modified from kvfran and ppo repository.
# -----------------------------------------------------------------------------
import init_path
from util import logger
from gated_graph_policy_network import GGNN
from baseline_network import tf_baseline_network


class tf_ggnn_baseline_network(GGNN, tf_baseline_network):

    def __init__(self,
                 session,
                 name_scope,
                 input_size,
                 placeholder_list,
                 weight_init_methods='orthogonal',
                 ob_placeholder=None,
                 trainable=True,
                 build_network_now=True,
                 args=None):

        root_connection_option = args.root_connection_option
        root_connection_option = root_connection_option.replace('Rn', 'Ra')
        root_connection_option = root_connection_option.replace('Rb', 'Ra')
        assert 'Rb' in root_connection_option or \
            'Ra' in root_connection_option, \
            logger.error(
                'Root connection option {} invalid for baseline'.format(
                    root_connection_option
                )
            )
        self._base_dir = init_path.get_abs_base_dir()

        GGNN.__init__(
            self,
            session=session,
            name_scope=name_scope,
            input_size=input_size,
            output_size=1,
            weight_init_methods=weight_init_methods,
            ob_placeholder=ob_placeholder,
            trainable=trainable,
            build_network_now=build_network_now,
            is_baseline=True,
            placeholder_list=placeholder_list,
            args=args
        )

        self._build_train_placeholders()
        self._build_loss()

    def get_vf_loss(self):
        return self._loss
