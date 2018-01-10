#!/usr/bin/env python2
# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang, Jun 23rd, 2017
# -----------------------------------------------------------------------------


import init_path
from util import logger
import mujoco_parser
import numpy as np


_BASE_DIR = init_path.get_base_dir()


def map_output(transfer_env, i_value, added_constant, gnn_option_list):
    '''
        @brief:
            i_value could be the logstd (1, num_action), policy_output/w
            (64, num_action), policy_output/b (1, num_action)
    '''
    assert len(gnn_option_list) == 4
    i_value = np.transpose(i_value)  # make the num_action to the front
    ienv, oenv = [env + '-v1' for env in transfer_env.split('2')]
    ienv_info = mujoco_parser.parse_mujoco_graph(
        ienv,
        gnn_node_option=gnn_option_list[0],
        root_connection_option=gnn_option_list[1],
        gnn_output_option=gnn_option_list[2],
        gnn_embedding_option=gnn_option_list[3]
    )
    oenv_info = mujoco_parser.parse_mujoco_graph(
        oenv,
        gnn_node_option=gnn_option_list[0],
        root_connection_option=gnn_option_list[1],
        gnn_output_option=gnn_option_list[2],
        gnn_embedding_option=gnn_option_list[3]
    )
    if len(i_value.shape) > 1:
        o_value = np.zeros([len(oenv_info['output_list']), i_value.shape[1]])
    else:
        # the b matrix
        o_value = np.zeros([len(oenv_info['output_list'])])
    assert len(i_value) == len(ienv_info['output_list'])

    ienv_node_name_list = [node['name'] for node in ienv_info['tree']]
    for output_id, output_node_id in enumerate(oenv_info['output_list']):
        # get the name of the joint
        node_name = oenv_info['tree'][output_node_id]['name']
        # if the node is alreay in the input environment?
        if node_name in ienv_node_name_list:
            if ienv_node_name_list.index(node_name) not in \
                    ienv_info['output_list']:
                logger.warning('Missing joint: {}'.format(node_name))
                continue
            o_value[output_id] = i_value[
                ienv_info['output_list'].index(
                    ienv_node_name_list.index(node_name)
                )
            ]
        else:
            # the name format: "@type_@name_@number", e.g.: joint_leg_1
            assert len(node_name.split('_')) == 3
            # find all the repetitive node and calculate the average
            repetitive_struct_node_list = [
                ienv_node_name_list.index(name)
                for name in ienv_node_name_list
                if node_name.split('_')[1] == name.split('_')[1]
            ]
            num_reptitive_nodes = float(len(repetitive_struct_node_list))
            assert len(repetitive_struct_node_list) >= 1

            for i_node_id in repetitive_struct_node_list:
                o_value[output_id] += i_value[
                    ienv_info['output_list'].index(i_node_id)
                ] / num_reptitive_nodes
    return np.transpose(o_value) + added_constant


def map_input(transfer_env, i_value, added_constant, gnn_option_list):
    assert len(gnn_option_list) == 4
    ienv, oenv = [env + '-v1' for env in transfer_env.split('2')]
    ienv_info = mujoco_parser.parse_mujoco_graph(
        ienv,
        gnn_node_option=gnn_option_list[0],
        root_connection_option=gnn_option_list[1],
        gnn_output_option=gnn_option_list[2],
        gnn_embedding_option=gnn_option_list[3]
    )
    oenv_info = mujoco_parser.parse_mujoco_graph(
        oenv,
        gnn_node_option=gnn_option_list[0],
        root_connection_option=gnn_option_list[1],
        gnn_output_option=gnn_option_list[2],
        gnn_embedding_option=gnn_option_list[3]
    )
    o_value = np.zeros([oenv_info['debug_info']['ob_size'], i_value.shape[1]])
    assert len(i_value) == ienv_info['debug_info']['ob_size']

    ienv_node_name_list = [node['name'] for node in ienv_info['tree']]
    for output_id, output_node_id in oenv_info['input_dict'].iteritems():
        # get the name of the joint
        node_name = oenv_info['tree'][output_id]['name']
        # if the node is alreay in the input environment?
        if node_name in ienv_node_name_list:
            o_value[output_node_id] = i_value[
                ienv_info['input_dict'][
                    ienv_node_name_list.index(node_name)
                ]
            ]
        else:
            continue
    return o_value


def map_transfer_env_running_mean(ienv, oenv, running_mean_info,
                                  observation_size,
                                  gnn_node_option, root_connection_option,
                                  gnn_output_option, gnn_embedding_option):

    # parse the mujoco information
    ienv_info = mujoco_parser.parse_mujoco_graph(
        ienv,
        gnn_node_option=gnn_node_option,
        root_connection_option=root_connection_option,
        gnn_output_option=gnn_output_option,
        gnn_embedding_option=gnn_embedding_option
    )
    oenv_info = mujoco_parser.parse_mujoco_graph(
        oenv,
        gnn_node_option=gnn_node_option,
        root_connection_option=root_connection_option,
        gnn_output_option=gnn_output_option,
        gnn_embedding_option=gnn_embedding_option
    )
    i_running_mean_info = running_mean_info
    # we start the running mean by cutting the mean to 0.1
    start_coeff = 1
    o_running_mean_info = {
        'step': i_running_mean_info['step'] * start_coeff,
        'mean': np.zeros([observation_size]),
        'variance': np.zeros([observation_size]),
        'square_sum': np.zeros([observation_size]),
        'sum': np.zeros([observation_size])
    }
    ienv_node_name_list = [node['name'] for node in ienv_info['tree']]

    for node, oenv_digit in oenv_info['input_dict'].iteritems():
        node_name = oenv_info['tree'][node]['name']
        # if the node is alreay in the input environment?
        if node_name in ienv_node_name_list:
            ienv_digit = ienv_info['input_dict'][
                ienv_node_name_list.index(node_name)
            ]
            assert len(ienv_digit) == len(oenv_digit)

            # assign the value!
            for key in ['square_sum', 'sum']:
                o_running_mean_info[key][oenv_digit] = \
                    i_running_mean_info[key][ienv_digit] * start_coeff
            for key in ['mean', 'variance']:
                o_running_mean_info[key][oenv_digit] = \
                    i_running_mean_info[key][ienv_digit]
        else:
            # the name format: "@type_@name_@number", e.g.: joint_leg_1
            assert len(node_name.split('_')) == 3
            # find all the repetitive node and calculate the average
            repetitive_struct_node_list = [
                ienv_node_name_list.index(name)
                for name in ienv_node_name_list
                if node_name.split('_')[1] == name.split('_')[1]
            ]
            assert len(repetitive_struct_node_list) >= 1
            num_reptitive_nodes = float(len(repetitive_struct_node_list))

            for i_node_id in repetitive_struct_node_list:
                ienv_digit = ienv_info['input_dict'][i_node_id]
                assert len(ienv_digit) == len(oenv_digit)
                # assign the value!
                for key in ['square_sum', 'sum']:
                    o_running_mean_info[key][oenv_digit] += \
                        i_running_mean_info[key][ienv_digit] * \
                        start_coeff / num_reptitive_nodes
                for key in ['mean', 'variance']:
                    o_running_mean_info[key][oenv_digit] += \
                        i_running_mean_info[key][ienv_digit] / \
                        num_reptitive_nodes

    return o_running_mean_info
