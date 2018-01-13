# ------------------------------------------------------------------------------
#   @brief:
#       record the parameters here
#   @author:
#       Tingwu Wang, 2017, June, 12th
# ------------------------------------------------------------------------------

import argparse
import init_path


def get_config():
    # get the parameters
    parser = argparse.ArgumentParser(description='graph_rl.')

    # the experiment settings
    parser.add_argument("--purpose", type=str, default='debug_experiments',
                        help='the name to be recorded to the experiments')
    parser.add_argument("--task", type=str, default='Reacher-v1',
                        help='the mujoco environment to test')
    parser.add_argument("--gamma", type=float, default=.99,
                        help='the discount factor for value function')
    parser.add_argument("--output_dir", '-o', type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)

    # training configuration
    parser.add_argument("--timesteps_per_batch", type=int, default=2050,
                        help='number of steps in the rollout')
    parser.add_argument("--max_timesteps", type=int, default=1e6)
    parser.add_argument("--advantage_method", type=str, default='gae',
                        help="['gae', 'raw']")
    parser.add_argument("--gae_lam", type=float, default=.95)
    parser.add_argument("--use_gpu", type=int, default=0,
                        help='1 for yes, 0 for no')

    # ppo configuration
    parser.add_argument("--num_threads", type=int, default=5)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value_lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip_value", type=float, default=5.0)
    parser.add_argument("--optim_epochs", type=int, default=10)
    parser.add_argument("--extra_vf_optim_epochs", type=int, default=0)
    parser.add_argument("--optim_batch_size", type=int, default=64)
    parser.add_argument("--minibatch_all_feed", type=int, default=0,
                        help='if set 1, batch_size = dataset_size')
    parser.add_argument("--target_kl", type=float, default=0.01)
    parser.add_argument("--target_kl_high", type=float, default=2)
    parser.add_argument("--target_kl_low", type=float, default=0.5)
    parser.add_argument("--use_weight_decay", type=int, default=0)
    parser.add_argument("--weight_decay_coeff", type=float, default=1e-5)

    # the checkpoint and summary setting
    parser.add_argument("--ckpt_name", type=str, default=None)
    parser.add_argument("--checkpoint_start_iteration", '-c',
                        type=int, default=50)
    parser.add_argument("--min_ckpt_iteration_diff", '-m', type=int, default=20)
    parser.add_argument("--summary_freq", type=int, default=1)
    parser.add_argument("--video_freq", type=int, default=2500)
    parser.add_argument("--time_id", type=str, default=init_path.get_time(),
                        help='the special id of the experiment')
    parser.add_argument("--checkpoint_ignore_varstr", type=str,
                        default="INVALID")

    # network settings
    parser.add_argument("--baseline_type", type=str, default='tf',
                        help="['np', 'tf']")
    parser.add_argument("--network_shape", type=str, default='64,64',
                        help='For the general policy network')

    # adaptive kl (we are not using it in the coming experiments)
    parser.add_argument("--use_kl_penalty", type=int, default=0)
    parser.add_argument("--kl_alpha", type=float, default=1.5)
    parser.add_argument("--kl_eta", type=float, default=50)

    # adaptive lr (maybe only necessary for the humanoid)
    parser.add_argument("--lr_schedule", type=str, default='linear',
                        help='["linear", "constant", "adaptive"]')
    parser.add_argument("--lr_alpha", type=int, default=2)

    # debug setting
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--write_log', type=int, default=1)
    parser.add_argument('--write_summary', type=int, default=1)
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument("--test", type=int, default=0,
                        help='if not 0, test for this number of episodes')

    # the settings for the ggnn
    parser = get_ggnn_config(parser)
    args = parser.parse_args()
    args = post_process(args)

    return args


def get_ggnn_config(parser):
    parser.add_argument("--use_gnn_as_policy", type=int, default=0,
                        help='0 for just the standard policy network, ' +
                        '1 for using the gnn')
    parser.add_argument("--use_gnn_as_value", type=int, default=0)

    parser.add_argument("--gnn_input_feat_dim", type=int, default=64)
    parser.add_argument("--gnn_node_hidden_dim", type=int, default=64)
    parser.add_argument("--gnn_num_prop_steps", type=int, default=4)
    parser.add_argument("--gnn_init_method", type=str, default='orthogonal')
    parser.add_argument("--gnn_node_option", type=str, default='nG,nB',
                        help='''
                            @brief:
                                @'nG' / 'yG':
                                    whether we allow 'geom' node in the graph
                                @'nB' / 'yB':
                                    whether we allow 'body' node in the graph
                        ''')
    parser.add_argument("--root_connection_option", type=str, default='nN,Rb,uE',
                        help='''
                        Change this parameter to change the graph.

                        @Options:
                            ['nN,Rn', 'nN,Rb', 'nN,Ra',
                             'yN,Rn', 'yN,Rb', 'yN,Ra'] + ['uE', 'sE']
                        @brief:
                            @'nN' / 'yN':
                                Whether we allow neighbour connection in the
                                model.
                            @'Rn' / 'Rb' / 'Ra':
                                No additional connection from the root.
                                Add connections from the root to 'body' node.
                                Add connections from the root to all nodes.
                            @'uE' / 'sE':
                                'uE': only one edge type will be used
                                'sE': different edge types will be used
                        ''')
    parser.add_argument("--gnn_output_option", type=str, default='unified',
                        help='''
                        @Options:
                            ["unified", "separate", "shared"]

                        @unified:
                            Only one output MLP is used for all output joint
                        @separate:
                            For every joint, a unique MLP is assigned to
                            generate the output torque.
                        @shared:
                            For separate type of nodes, we have different MLP.
                            For example, the left thigh joint and right thigh
                            joint share the same output mlp
                        ''')
    parser.add_argument("--gnn_embedding_option", type=str, default='shared',
                        help='''
                        @Options:
                            ["parameter", "shared", "noninput_separate",
                             "noninput_shared"]

                        @parameter:
                            Embedding input is the node parameter vector
                        @shared:
                            Embedding input is the one-hot encoding. For nodes
                            with the same structure position, e.g. left thigh
                            and right thigh, we provide shared encoding.
                        @separate:
                            Embedding input is the one-hot encoding. For each
                            node, we provide separate encoding.
                        @noninput_separate:
                            Embedding input is just a gather index to select
                            the variable, every input is different
                        @noninput_shared:
                            Embedding input is just a gather index to select
                            the variable, shared embedding is used
                        ''')
    parser.add_argument("--shared_network", type=int, default=0)
    parser.add_argument("--baseline_loss_coeff", type=float, default=1.0)
    parser.add_argument("--node_update_method", type=str, default='GRU',
                        help='could be either GRU or MLP update')
    parser.add_argument("--transfer_env", type=str,
                        default='Nothing2Nothing',
                        help='''
                        the pretrained env and the new env name, for example, we
                        can use "SnakeFour2SnakeThree"
                        ''')
    parser.add_argument("--logstd_option", type=str,
                        default='load',
                        help='''
                            @"load":
                                load the logstd as what it should be
                            @"load_float":
                                load logstd, but add a constant (bigger
                                exploration)
                            @"fix_float":
                                do not load logstd, load a constant (bigger
                                exploration)
                        ''')
    parser.add_argument('--mlp_raw_transfer', type=int, default=0,
                        help='''
                        if set to 1, we just load the raw mlp transfer
                        So we have mlpt, mlpr, and ggnn for transfer learning
                        if set to 2, we skip the unmatched weights
                        ''')
    return parser


def post_process(args):
    if args.debug:
        args.write_log = 0
        args.write_summary = 0
        args.monitor = 0

    # parse the network shape
    args.network_shape = [
        int(hidden_size) for hidden_size in args.network_shape.split(',')
    ]
    args.task_name = args.task

    return args
