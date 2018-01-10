#!/usr/bin/env python2
# -----------------------------------------------------------------------------
#   @brief:
#       In this function, we change the data from [batch_size, ob_dim] into
#       [batch_size * num_node, hidden_dim]
#   @author:
#       Tingwu Wang, Jul. 13th, 2017
# -----------------------------------------------------------------------------


import numpy as np
import init_path
from util import logger
from six.moves import xrange

_ABS_BASE_PATH = init_path.get_abs_base_dir()


def construct_graph_input_feeddict(node_info,
                                   obs_n,
                                   receive_idx,
                                   send_idx,
                                   node_type_idx,
                                   inverse_node_type_idx,
                                   output_type_idx,
                                   inverse_output_type_idx,
                                   last_batch_size,
                                   request_data=['ob', 'idx']):
    '''
        @brief:
            @obs_n: the observation in the [batch_size * node, hidden_size]

            @send_idx: [shape=None for each edge agent_type]

            @node_type_idx: [shape=None for each edge agent_type]

            @receive_idx: [shape=None], the number of element in accordance
                with the send_idx

            @inverse_node_type_idx: [shape=None], the number of element in
                accordance with the node_type_idx

            @last_batch_size: if the batch is the same, no need to calculate
                again
    '''

    assert len(request_data) > 0 and \
        np.all([dtype in ['ob', 'idx'] for dtype in request_data]), \
        logger.error('at least one of the ob and index is calculated')
    graph_ob = -1
    graph_parameters = -1
    current_batch_size = -1

    # reconstruct the giant receive_idx and send_idx list if needed
    if 'idx' in request_data:
        current_batch_size = obs_n.shape[0]
        if not current_batch_size == last_batch_size:
            receive_idx, send_idx, \
                node_type_idx, inverse_node_type_idx, \
                output_type_idx, inverse_output_type_idx = \
                _construct_index(node_info, current_batch_size)

    if 'ob' in request_data:
        # get the obs from [batch_size, hidden_size] into [batch_size *
        # node_size, hidden_size]
        current_batch_size = obs_n.shape[0]
        graph_ob, graph_parameters = _get_obs(node_info,
                                              obs_n,
                                              current_batch_size)

    return graph_ob, graph_parameters, \
        receive_idx, send_idx, \
        node_type_idx, inverse_node_type_idx, \
        output_type_idx, inverse_output_type_idx, \
        current_batch_size


def _construct_index(node_info, batch_size):
    '''
        @brief:
            get the receive index and the send index for the network as feed
            dict
    '''
    logger.info('Building new send_idx and receive_idx for the model, ' +
                'current batch size: {}'.format(batch_size))
    # size [batch_size]
    offset = node_info['num_nodes'] * np.array(xrange(batch_size))

    # STEP 1: get the receive index and inverse_node_type index
    # the receive_idx is now [batch_size * num_edges], 'batch' comes first
    receive_idx_raw = {}
    receive_idx = []
    for i_edge_type in node_info['edge_type_list']:
        # NOTE: the order in node_info['edge_type_list'] matters
        receive_idx_raw[i_edge_type] = _add_offset(
            node_info['receive_idx_raw'][i_edge_type],
            offset,
            batch_size,
            if_reshape=True
        )
        receive_idx.extend(receive_idx_raw[i_edge_type])

    # STEP 2: get the send index
    # the send_idx is a list of np.arrays [np.array for i_edge in edge_list]
    send_idx = {}
    for i_edge_type in node_info['edge_type_list']:
        send_idx[i_edge_type] = _add_offset(node_info['send_idx'][i_edge_type],
                                            offset,
                                            batch_size,
                                            if_reshape=True)

    # STEP 3: get the inverse node idx after ordered by type
    inverse_node_type_idx = _add_inverse_offset(
        node_info['inverse_node_extype_offset'],
        node_info['inverse_node_intype_offset'],
        node_info['inverse_node_self_offset'],
        batch_size
    )

    # STEP 4: get the inverse output idx after ordered by type
    inverse_output_type_idx = _add_inverse_offset(
        node_info['inverse_output_extype_offset'],
        node_info['inverse_output_intype_offset'],
        node_info['inverse_output_self_offset'],
        batch_size
    )

    # STEP 5: get the node type dict
    node_type_idx = {}
    for i_node_type in node_info['node_type_dict']:
        node_type_idx[i_node_type] = _add_offset(
            node_info['node_type_dict'][i_node_type],
            offset,
            batch_size,
            if_reshape=True
        )

    # STEP 6: get the output type dict
    output_type_idx = {}
    for i_output_type in node_info['output_type_dict']:
        output_type_idx[i_output_type] = _add_offset(
            node_info['output_type_dict'][i_output_type],
            offset,
            batch_size,
            if_reshape=True
        )

    # import util.fpdb as fpdb; fpdb.fpdb().set_trace()
    return receive_idx, send_idx, node_type_idx, \
        inverse_node_type_idx, output_type_idx, inverse_output_type_idx


def _get_obs(node_info, raw_ob, batch_size):
    '''
        @brief:
            In this function, we change the data from [batch_size, ob_dim] into
            [batch_size * num_node, hidden_dim]

        @input:
            @raw_ob ("obs_n"): [batch_size, ob_dim]

        @output:
            @graph_ob:
                a dictionary,
                graph_ob['geom']: [batch_size, node_ob_size]

            [batch_size * node_size, hidden_dim], padded by zeros
    '''
    # step 0: init the variables
    graph_ob = {
        node_type: np.zeros(
            [batch_size,
                len(node_info['node_type_dict'][node_type]),
                abs(node_info['ob_size_dict'][node_type])]
        )
        for node_type in node_info['node_type_dict']
    }
    # NOTE: compatible for the tree network
    graph_parameters = {
        node_type: np.zeros(
            [batch_size,
                len(node_info['node_type_dict'][node_type]),
                abs(node_info['para_size_dict'][node_type])]
        )
        for node_type in node_info['node_type_dict']
    }

    # step 1: construct the graph obs. For each node, fetch the input for it.
    for node_type in node_info['node_type_dict']:
        # 'geom', 'joint', 'root', 'body', we already know that the size of ob
        # for one type is consistent
        if node_info['ob_size_dict'][node_type] <= 0:
            continue  # this type do not receive any ob, leave them zeros

        for node_pos_in_graph_ob, i_node_id in enumerate(
                node_info['node_type_dict'][node_type]):
            # NOTE: WARNING: THIS IS NOT GOOD ACTUALLY
            # graph_ob[node_type][:, node_pos_in_graph_ob, :] = \
            #     raw_ob[:, node_info['input_dict'][i_node_id]]
            graph_ob[node_type][:, node_pos_in_graph_ob,
                                :len(node_info['input_dict'][i_node_id])] = \
                raw_ob[:, node_info['input_dict'][i_node_id]]

    # step 2: construct the graph parameters
    for node_type in node_info['node_type_dict']:
        # similar to the graph obs' construction
        graph_parameters[node_type][:, :, :] = np.tile(
            node_info['node_parameters'][node_type], [batch_size, 1, 1]
        )

    # step 3: reshape the graph_ob and the graph_parameters
    for node_type in node_info['node_type_dict']:
        if node_info['ob_size_dict'][node_type] == 0:
            this_shape = graph_ob[node_type].shape
            assert this_shape[2] == 0, logger.error('wrong ob shape!')
            graph_ob[node_type] = np.zeros([this_shape[0] * this_shape[1], 0])
        else:
            graph_ob[node_type] = graph_ob[node_type].reshape(
                [-1, abs(node_info['ob_size_dict'][node_type])]
            )

        if node_info['para_size_dict'][node_type] == 0:
            this_shape = graph_parameters[node_type].shape
            assert this_shape[2] == 0, logger.error('wrong ob shape!')
            graph_parameters[node_type] = np.zeros(
                [this_shape[0] * this_shape[1], 0]
            )
        else:
            graph_parameters[node_type] = graph_parameters[node_type].reshape(
                [-1, node_info['para_size_dict'][node_type]]
            )

    return graph_ob, graph_parameters


def _add_offset(idx, offset, batch_size, if_reshape):
    '''
        @brief: add the offset to the index
    '''
    # size: [batch_size, original_size]
    tiled_idx = np.tile(idx, [batch_size, 1])
    tiled_idx = \
        tiled_idx + (np.tile(offset, [tiled_idx.shape[1], 1])).transpose()

    if if_reshape:
        tiled_idx = tiled_idx.reshape([-1])
    return tiled_idx


def _add_inverse_offset(extype_offset, intype_offset, self_offset, batch_size):
    # construct the first batch's index
    first_index = batch_size * extype_offset + intype_offset
    index = np.tile(first_index, [batch_size, 1])

    # the offset
    incremental_offset = np.tile(self_offset, [batch_size, 1])
    incremental_offset *= np.expand_dims(range(batch_size), 1)

    index = index + incremental_offset
    index = index.reshape([-1])

    return index
