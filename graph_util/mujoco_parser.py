#!/usr/bin/env python2
# -----------------------------------------------------------------------------
#   @brief:
#       Some helper functions to parse the mujoco xml template files
#   @author:
#       Tingwu Wang, Jun 23rd, 2017
#   @UPDATE:
#       1. add the tendon support
#       2. add support for HumanoidStandup, halfCheeta, swimmer, hopper, walker
#       3. add support for reacher, pendulumm
#       4. MAJOR UPDATE Aug. 21, 2017: now the root only absorb the joints that
#           are not motors.
#       5. MAJOR UPDATE, removing all the geom node, Sept. 10th, 2017
# -----------------------------------------------------------------------------


import init_path
import os
import numpy as np
from bs4 import BeautifulSoup as bs
from util import logger
from environments import register


__all__ = ['parse_mujoco_graph']

XML_ASSERT_DIR = os.path.join(init_path.get_base_dir(),
                              'environments',
                              'assets')

'''
    Definition of nodes:
    @root:
        The 'root' type is the combination of the top level 'body' node and
        the top level free 'joint' (two nodes combined)
        Also, additional input will be assigned to the root node
        (e.g. the postion of the targer).

        For different tasks, we should have different MLP for each root.

    @geom, @body, @joint:
        The structure defined in the xml files. Ideally, the MLP for input,
        propogation, and output could be shared among different models.
'''
_NODE_TYPE = ['root', 'joint', 'body']  # geom is removed
EDGE_TYPE = {'self_loop': 0, 'root-root': 0,  # root-root is loop, also 0
             'joint-joint': 1, 'geom-geom': 2, 'body-body': 3, 'tendon': 4,
             'joint-geom': 5, 'geom-joint': -5,  # pc-relationship
             'joint-body': 6, 'body-joint': -6,
             'body-geom': 7, 'geom-body': -7,
             'root-geom': 8, 'geom-root': -8,
             'root-joint': 9, 'joint-root': -9,
             'root-body': 10, 'body-root': -10}

SYMMETRY_MAP, XML_DICT, OB_MAP, JOINT_KEY, ROOT_OB_SIZE, BODY_KEY = \
    register.get_mujoco_model_settings()

# development status
ALLOWED_JOINT_TYPE = ['hinge', 'free', 'slide']

MULTI_TASK_DICT = register.MULTI_TASK_DICT


def parse_mujoco_graph(task_name,
                       gnn_node_option='nG,yB',
                       xml_path=None,
                       root_connection_option='nN, Rn, sE',
                       gnn_output_option='shared',
                       gnn_embedding_option='shared'):
    '''
        @brief:
            get the tree of "geom", "body", "joint" built up.

        @return:
            @tree: This function will return a list of dicts. Every dict
                contains the information of node.

                tree[id_of_the_node]['name']: the unique identifier of the node

                tree[id_of_the_node]['neighbour']: is the list for the
                neighbours

                tree[id_of_the_node]['type']: could be 'geom', 'body' or
                'joint'

                tree[id_of_the_node]['info']: debug info from the xml. it
                should not be used during model-free optimization

                tree[id_of_the_node]['is_output_node']: True or False

            @input_dict: input_dict['id_of_the_node'] = [position of the ob]

            @output_list: This correspond to the id of the node where a output
                is available
    '''
    if xml_path is None:  # the xml file is in environments/assets/
        xml_path = os.path.join(XML_ASSERT_DIR, XML_DICT[task_name])

    infile = open(xml_path, 'r')
    xml_soup = bs(infile.read(), 'xml')
    if 'nG' in gnn_node_option:
        # no geom node allowed, this order is very important, 'body' must be
        # after the 'joint'
        node_type_allowed = ['root', 'joint', 'body']
    else:
        assert 'yG' in gnn_node_option
        node_type_allowed = ['root', 'joint', 'body', 'geom']
        logger.warning('Using Geom is not a good idea!!\n\n\n')

    # step 0: get the basic information of the nodes ready
    tree, node_type_dict = _get_tree_structure(xml_soup, node_type_allowed)

    # step 1: get the neighbours and relation tree
    tree, relation_matrix = _append_tree_relation(tree,
                                                  node_type_allowed,
                                                  root_connection_option)
    # step 2: get the tendon relationship ready
    tree, relation_matrix = _append_tendon_relation(tree,
                                                    relation_matrix,
                                                    xml_soup)

    # step 3: get the input list ready
    input_dict, ob_size = _get_input_info(tree, task_name)

    # step 4: get the output list ready
    tree, output_list, output_type_dict, action_size = \
        _get_output_info(tree, xml_soup, gnn_output_option)

    # step 5: get the node parameters
    node_parameters, para_size_dict = _append_node_parameters(
        tree, xml_soup, node_type_allowed, gnn_embedding_option
    )

    debug_info = {'ob_size': ob_size, 'action_size': action_size}

    # step 6: prune the body nodes
    if 'nB' in gnn_node_option:
        tree, relation_matrix, node_type_dict, \
            input_dict, node_parameters, para_size_dict = \
            _prune_body_nodes(tree=tree,
                              relation_matrix=relation_matrix,
                              node_type_dict=node_type_dict,
                              input_dict=input_dict,
                              node_parameters=node_parameters,
                              para_size_dict=para_size_dict,
                              root_connection_option=root_connection_option)

    # step 7: (optional) uni edge type?
    if 'uE' in root_connection_option:
        relation_matrix[np.where(relation_matrix != 0)] = 1
    else:
        assert 'sE' in root_connection_option

    return dict(tree=tree,
                relation_matrix=relation_matrix,
                node_type_dict=node_type_dict,
                output_type_dict=output_type_dict,
                input_dict=input_dict,
                output_list=output_list,
                debug_info=debug_info,
                node_parameters=node_parameters,
                para_size_dict=para_size_dict,
                num_nodes=len(tree))


def _prune_body_nodes(tree,
                      relation_matrix,
                      node_type_dict,
                      input_dict,
                      node_parameters,
                      para_size_dict,
                      root_connection_option):
    '''
        @brief:
            In this function, we will have to remove the body node.
            1. delete all the the bodys, whose ob will be placed into its kid
                joint (multiple joints possible)
            2. for the root node, if kid joint exists, transfer the ownership
                body ob into the kids
    '''
    # make sure the tree is structured in a 'root', 'joint', 'body' order
    assert node_type_dict['root'] == [0] and \
        max(node_type_dict['joint']) < min(node_type_dict['body']) and \
        'geom' not in node_type_dict

    # for each joint, let it eat its father body root, the relation_matrix and
    # input_dict need to be inherited
    for node_id, i_node in enumerate(tree[0: min(node_type_dict['body'])]):
        if i_node['type'] is not 'joint':
            assert i_node['type'] is 'root'
            continue

        # find all the parent
        parent = i_node['parent']

        # inherit the input observation
        if parent in input_dict:
            input_dict[node_id] += input_dict[parent]
        '''
            1. inherit the joint with shared body, only the first joint will
                inherit the AB_body's relationship. other joints will be
                attached to the first joint
                A_joint ---- AB_body ---- B_joint. On
            2. inherit joint-joint relationships for sybling joints:
                A_joint ---- A_body ---- B_body ---- B_joint
            3. inherit the root-joint connection
                A_joint ---- A_body ---- root
        '''
        # step 1: check if there is brothers / sisters of this node
        children = np.where(
            relation_matrix[parent, :] == EDGE_TYPE['body-joint']
        )[0]
        first_brother = [child_id for child_id in children
                         if child_id != node_id and child_id < node_id]
        if len(first_brother) > 0:
            first_brother = min(first_brother)
            relation_matrix[node_id, first_brother] = EDGE_TYPE['joint-joint']
            relation_matrix[first_brother, node_id] = EDGE_TYPE['joint-joint']
            continue

        # step 2: the type 2 relationship, note that only the first brother
        # will be considered
        uncles = np.where(
            relation_matrix[parent, :] == EDGE_TYPE['body-body']
        )[0]
        for i_uncle in uncles:
            syblings = np.where(
                relation_matrix[i_uncle, :] == EDGE_TYPE['body-joint']
            )[0]
            if len(syblings) > 0:
                sybling = syblings[0]
            else:
                continue
            if tree[sybling]['parent'] is tree[i_uncle]['parent']:
                continue
            relation_matrix[node_id, sybling] = EDGE_TYPE['joint-joint']
            relation_matrix[sybling, node_id] = EDGE_TYPE['joint-joint']

        # step 3: the type 3 relationship
        uncles = np.where(
            relation_matrix[parent, :] == EDGE_TYPE['body-root']
        )[0]
        assert len(uncles) <= 1
        for i_uncle in uncles:
            relation_matrix[node_id, i_uncle] = EDGE_TYPE['joint-root']
            relation_matrix[i_uncle, node_id] = EDGE_TYPE['root-joint']

    # remove all the body root
    first_body_node = min(node_type_dict['body'])
    tree = tree[:first_body_node]
    relation_matrix = relation_matrix[:first_body_node, :first_body_node]
    for i_body_node in node_type_dict['body']:
        if i_body_node in input_dict:
            input_dict.pop(i_body_node)
    node_parameters.pop('body')
    node_type_dict.pop('body')
    para_size_dict.pop('body')

    for i_node in node_type_dict['joint']:
        assert len(input_dict[i_node]) == len(input_dict[1])

    return tree, relation_matrix, node_type_dict, \
        input_dict, node_parameters, para_size_dict


def _get_tree_structure(xml_soup, node_type_allowed):
    mj_soup = xml_soup.find('worldbody').find('body')
    tree = []  # NOTE: the order in the list matters!
    tree_id = 0
    nodes = dict()

    # step 0: set the root node
    motor_names = _get_motor_names(xml_soup)
    node_info = {'type': 'root', 'is_output_node': False,
                 'name': 'root_mujocoroot',
                 'neighbour': [], 'id': 0,
                 'info': mj_soup.attrs, 'raw': mj_soup}
    node_info['attached_joint_name'] = \
        [i_joint['name']
         for i_joint in mj_soup.find_all('joint', recursive=False)
         if i_joint['name'] not in motor_names]
    for key in JOINT_KEY:
        node_info[key + '_size'] = 0
    node_info['attached_joint_info'] = []
    node_info['tendon_nodes'] = []
    tree.append(node_info)
    tree_id += 1

    # step 1: set the 'node_type_allowed' nodes
    for i_type in node_type_allowed:
        nodes[i_type] = mj_soup.find_all(i_type)
        if i_type == 'body':
            # the root body should be added to the body
            nodes[i_type] = [mj_soup] + nodes[i_type]
        if len(nodes[i_type]) == 0:
            continue
        for i_node in nodes[i_type]:
            node_info = dict()
            node_info['type'] = i_type
            node_info['is_output_node'] = False
            node_info['raw_name'] = i_node['name']
            node_info['name'] = node_info['type'] + '_' + i_node['name']
            node_info['tendon_nodes'] = []
            # this id is the same as the order in the tree
            node_info['id'] = tree_id
            node_info['parent'] = None

            # additional debug information, should not be used during training
            node_info['info'] = i_node.attrs
            node_info['raw'] = i_node

            # NOTE: the get the information about the joint that is directly
            # attached to 'root' node. These joints will be merged into the
            # 'root' node
            if i_type == 'joint' and \
                    i_node['name'] in tree[0]['attached_joint_name']:
                tree[0]['attached_joint_info'].append(node_info)
                for key in JOINT_KEY:
                    tree[0][key + '_size'] += ROOT_OB_SIZE[key][i_node['type']]
                continue

            # currently, only 'hinge' type is supported
            if i_type == 'joint' and i_node['type'] not in ALLOWED_JOINT_TYPE:
                logger.warning(
                    'NOT IMPLEMENTED JOINT TYPE: {}'.format(i_node['type'])
                )

            tree.append(node_info)
            tree_id += 1

        logger.info('{} {} found'.format(len(nodes[i_type]), i_type))
    node_type_dict = {}

    # step 2: get the node_type dict ready
    for i_key in node_type_allowed:
        node_type_dict[i_key] = [i_node['id'] for i_node in tree
                                 if i_key == i_node['type']]
        assert len(node_type_dict) >= 1, logger.error(
            'Missing node type {}'.format(i_key))
    return tree, node_type_dict


def _append_tree_relation(tree, node_type_allowed, root_connection_option):
    '''
        @brief:
            build the relationship matrix and append relationship attribute
            to the nodes of the tree

        @input:
            @root_connection_option:
                'nN, Rn': without neighbour, no additional connection
                'nN, Rb': without neighbour, root connected to all body
                'nN, Ra': without neighbour, root connected to all node
                'yN, Rn': with neighbour, no additional connection
    '''
    num_node = len(tree)
    relation_matrix = np.zeros([num_node, num_node], dtype=np.int)

    # step 1: set graph connection relationship
    for i_node in tree:
        # step 1.1: get the id of the children
        children = i_node['raw'].find_all(recursive=False)
        if len(children) == 0:
            continue
        children_names = [i_children.name + '_' + i_children['name']
                          for i_children in children
                          if i_children.name in node_type_allowed]
        children_id_list = [
            [node['id'] for node in tree if node['name'] == i_children_name]
            for i_children_name in children_names
        ]

        i_node['children_id_list'] = children_id_list = \
            sum(children_id_list, [])  # squeeze the list
        current_id = i_node['id']
        current_type = tree[current_id]['type']

        # step 1.2: set the children-parent relationship edges
        for i_children_id in i_node['children_id_list']:
            relation_matrix[current_id, i_children_id] = \
                EDGE_TYPE[current_type + '-' + tree[i_children_id]['type']]
            relation_matrix[i_children_id, current_id] = \
                EDGE_TYPE[tree[i_children_id]['type'] + '-' + current_type]
            if tree[current_id]['type'] == 'body':
                tree[i_children_id]['parent'] = current_id

        # step 1.3 (optional): set children connected if needed
        if 'yN' in root_connection_option:
            for i_node_in_use_1 in i_node['children_id_list']:
                for i_node_in_use_2 in i_node['children_id_list']:
                    relation_matrix[i_node_in_use_1, i_node_in_use_2] = \
                        EDGE_TYPE[tree[i_node_in_use_1]['type'] + '-' +
                                  tree[i_node_in_use_2]['type']]

        else:
            assert 'nN' in root_connection_option, logger.error(
                'Unrecognized root_connection_option: {}'.format(
                    root_connection_option
                )
            )

    # step 2: set root connection
    if 'Ra' in root_connection_option:
        # if root is connected to all the nodes
        for i_node_in_use_1 in range(len(tree)):
            target_node_type = tree[i_node_in_use_1]['type']

            # add connections between all nodes and root
            relation_matrix[0, i_node_in_use_1] = \
                EDGE_TYPE['root' + '-' + target_node_type]
            relation_matrix[i_node_in_use_1, 0] = \
                EDGE_TYPE[target_node_type + '-' + 'root']

    elif 'Rb' in root_connection_option:
        for i_node_in_use_1 in range(len(tree)):
            target_node_type = tree[i_node_in_use_1]['type']

            if not target_node_type == 'body':
                continue

            # add connections between body and root
            relation_matrix[0, i_node_in_use_1] = \
                EDGE_TYPE['root' + '-' + target_node_type]
            relation_matrix[i_node_in_use_1, 0] = \
                EDGE_TYPE[target_node_type + '-' + 'root']
    else:
        assert 'Rn' in root_connection_option, logger.error(
            'Unrecognized root_connection_option: {}'.format(
                root_connection_option
            )
        )

    # step 3: unset the diagonal terms back to 'self-loop'
    np.fill_diagonal(relation_matrix, EDGE_TYPE['self_loop'])

    return tree, relation_matrix


def _append_tendon_relation(tree, relation_matrix, xml_soup):
    '''
        @brief:
            build the relationship of tendon (the spring)
    '''
    tendon = xml_soup.find('tendon')
    if tendon is None:
        return tree, relation_matrix
    tendon_list = tendon.find_all('fixed')

    for i_tendon in tendon_list:
        # find the id
        joint_name = ['joint_' + joint['joint']
                      for joint in i_tendon.find_all('joint')]
        joint_id = [node['id'] for node in tree if node['name'] in joint_name]
        assert len(joint_id) == 2, logger.error(
            'Unsupported tendon: {}'.format(i_tendon))

        # update the tree and the relationship matrix
        relation_matrix[joint_id[0], joint_id[1]] = EDGE_TYPE['tendon']
        relation_matrix[joint_id[1], joint_id[0]] = EDGE_TYPE['tendon']
        tree[joint_id[0]]['tendon_nodes'].append(joint_id[1])
        tree[joint_id[1]]['tendon_nodes'].append(joint_id[0])

        logger.info(
            'new tendon found between: {} and {}'.format(
                tree[joint_id[0]]['name'], tree[joint_id[1]]['name']
            )
        )

    return tree, relation_matrix


def _get_input_info(tree, task_name):
    input_dict = {}

    joint_id = [node['id'] for node in tree if node['type'] == 'joint']
    body_id = [node['id'] for node in tree if node['type'] == 'body']
    root_id = [node['id'] for node in tree if node['type'] == 'root'][0]

    # init the input dict
    input_dict[root_id] = []
    if 'cinert' in OB_MAP[task_name] or \
            'cvel' in OB_MAP[task_name] or 'cfrc' in OB_MAP[task_name]:
        candidate_id = joint_id + body_id
    else:
        candidate_id = joint_id
    for i_id in candidate_id:
        input_dict[i_id] = []

    logger.info('scanning ob information...')
    current_ob_id = 0
    for ob_type in OB_MAP[task_name]:

        if ob_type in JOINT_KEY:
            # step 1: collect the root ob's information. Some ob might be
            # ignore, which is specify in the SYMMETRY_MAP
            ob_step = tree[0][ob_type + '_size'] - \
                (ob_type == 'qpos') * SYMMETRY_MAP[task_name]
            input_dict[root_id].extend(
                range(current_ob_id, current_ob_id + ob_step)
            )
            current_ob_id += ob_step

            # step 2: collect the joint ob's information
            for i_id in joint_id:
                input_dict[i_id].append(current_ob_id)
                current_ob_id += 1

        elif ob_type in BODY_KEY:

            BODY_OB_SIZE = 10 if ob_type == 'cinert' else 6

            # step 0: skip the 'world' body
            current_ob_id += BODY_OB_SIZE

            # step 1: collect the root ob's information, note that the body will
            # still take this ob
            input_dict[root_id].extend(
                range(current_ob_id, current_ob_id + BODY_OB_SIZE)
            )
            # current_ob_id += BODY_OB_SIZE

            # step 2: collect the body ob's information
            for i_id in body_id:
                input_dict[i_id].extend(
                    range(current_ob_id, current_ob_id + BODY_OB_SIZE)
                )
                current_ob_id += BODY_OB_SIZE
        else:
            assert 'add' in ob_type, \
                logger.error('TYPE {BODY_KEY} NOT RECGNIZED'.format(ob_type))
            addition_ob_size = int(ob_type.split('_')[-1])
            input_dict[root_id].extend(
                range(current_ob_id, current_ob_id + addition_ob_size)
            )
            current_ob_id += addition_ob_size
        logger.info(
            'after {}, the ob size is reaching {}'.format(
                ob_type, current_ob_id
            )
        )
    return input_dict, current_ob_id  # to debug if the ob size is matched


def _get_output_info(tree, xml_soup, gnn_output_option):
    output_list = []
    output_type_dict = {}
    motors = xml_soup.find('actuator').find_all('motor')

    for i_motor in motors:
        joint_id = [i_node['id'] for i_node in tree
                    if 'joint_' + i_motor['joint'] == i_node['name']]
        if len(joint_id) == 0:
            # joint_id = 0  # it must be the root if not found
            assert False, logger.error(
                'Motor {} not found!'.format(i_motor['joint'])
            )
        else:
            joint_id = joint_id[0]
        tree[joint_id]['is_output_node'] = True
        output_list.append(joint_id)

        # construct the output_type_dict
        if gnn_output_option == 'shared':
            motor_type_name = i_motor['joint'].split('_')[0]
            if motor_type_name in output_type_dict:
                output_type_dict[motor_type_name].append(joint_id)
            else:
                output_type_dict[motor_type_name] = [joint_id]
        elif gnn_output_option == 'separate':
            motor_type_name = i_motor['joint']
            output_type_dict[motor_type_name] = [joint_id]
        else:
            assert gnn_output_option == 'unified', logger.error(
                'Invalid output type: {}'.format(gnn_output_option)
            )
            if 'unified' in output_type_dict:
                output_type_dict['unified'].append(joint_id)
            else:
                output_type_dict['unified'] = [joint_id]

    return tree, output_list, output_type_dict, len(motors)


def _get_motor_names(xml_soup):
    motors = xml_soup.find('actuator').find_all('motor')
    name_list = [i_motor['joint'] for i_motor in motors]
    return name_list


GEOM_TYPE_ENCODE = {
    'capsule': [0.0, 1.0],
    'sphere': [1.0, 0.0],
}


def _append_node_parameters(tree,
                            xml_soup,
                            node_type_allowed,
                            gnn_embedding_option):
    '''
        @brief:
            the output of this function is a dictionary.
        @output:
            e.g.: node_parameters['geom'] is a numpy array, which has the shape
            of (num_nodes, para_size_of_'geom')
            the node is ordered in the relative position in the tree
    '''
    assert node_type_allowed.index('joint') < node_type_allowed.index('body')

    if gnn_embedding_option == 'parameter':
        # step 0: get the para list and default setting for this mujoco xml
        PARAMETERS_LIST, default_dict = _get_para_list(xml_soup,
                                                       node_type_allowed)

        # step 2: get the node_parameter_list ready, they are in the node_order
        node_parameters = {node_type: [] for node_type in node_type_allowed}
        for node_id in range(len(tree)):
            output_parameter = []

            for i_parameter_type in PARAMETERS_LIST[tree[node_id]['type']]:
                # collect the information one by one
                output_parameter = _collect_parameter_info(
                    output_parameter, i_parameter_type,
                    tree[node_id]['type'], default_dict, tree[node_id]['info']
                )

            # this node is finished
            node_parameters[tree[node_id]['type']].append(output_parameter)

        # step 3: numpy the elements, and do validation check
        for node_type in node_type_allowed:
            node_parameters[node_type] = np.array(node_parameters[node_type],
                                                  dtype=np.float32)

        # step 4: get the size of parameters logged
        para_size_dict = {
            node_type: len(node_parameters[node_type][0])
            for node_type in node_type_allowed
        }

        # step 5: trick, root para is going to receive a dummy para [1]
        para_size_dict['root'] = 1
        node_parameters['root'] = np.ones([1, 1])
    elif gnn_embedding_option in \
            ['shared', 'noninput_separate', 'noninput_shared']:
        # step 1: preprocess, register the node, get the number of bits for
        # encoding needed
        struct_name_list = {node_type: [] for node_type in node_type_allowed}
        for node_id in range(len(tree)):
            name = tree[node_id]['name'].split('_')
            type_name = name[0]

            if gnn_embedding_option in ['noninput_separate']:
                register_name = name
                struct_name_list[type_name].append(register_name)
            else:  # shared
                register_name = type_name + '_' + name[1]
                if register_name not in struct_name_list[type_name]:
                    struct_name_list[type_name].append(register_name)
            tree[node_id]['register_embedding_name'] = register_name

        struct_name_list['root'] = [tree[0]['name']]  # the root
        tree[0]['register_embedding_name'] = tree[0]['name']

        # step 2: estimate the encoding length
        num_type_bits = 2
        para_size_dict = {  # 2 bits for type encoding
            i_node_type: num_type_bits + 8
            for i_node_type in node_type_allowed
        }

        # step 3: get the parameters
        node_parameters = {i_node_type: []
                           for i_node_type in node_type_allowed}
        appear_str = []
        for node_id in range(len(tree)):
            type_name = tree[node_id]['type']
            type_str = str(bin(node_type_allowed.index(type_name)))
            type_str = (type_str[2:]).zfill(num_type_bits)
            node_str = str(bin(struct_name_list[type_name].index(
                tree[node_id]['register_embedding_name']
            )))
            node_str = (node_str[2:]).zfill(
                para_size_dict[tree[node_id]['type']] - 2
            )

            if node_id == 0 or para_size_dict[type_name] == 2:
                node_str = ''

            final_str = type_str + node_str
            if final_str not in appear_str:
                appear_str.append(final_str)

            if 'noninput_shared_multi' in gnn_embedding_option:
                node_parameters[type_name].append(
                    tree[node_id]['register_embedding_name']
                )
            elif 'noninput' in gnn_embedding_option:
                node_parameters[type_name].append([appear_str.index(final_str)])
            else:
                node_parameters[type_name].append(
                    [int(i_char) for i_char in final_str]
                )

        # step 4: numpy the elements, and do validation check
        if gnn_embedding_option is not 'noninput_shared_multi':
            para_dtype = np.float32 \
                if gnn_embedding_option in ['parameter', 'shared'] \
                else np.int32
            for node_type in node_type_allowed:
                node_parameters[node_type] = \
                    np.array(node_parameters[node_type], dtype=para_dtype)
    else:
        assert False, logger.error(
            'Invalid option: {}'.format(gnn_embedding_option)
        )

    # step 5: postprocess
    # NOTE: make the length of the parameters the same
    if gnn_embedding_option in ['parameter', 'shared']:
        max_length = max([para_size_dict[node_type]
                         for node_type in node_type_allowed])
        for node_type in node_type_allowed:
            shape = node_parameters[node_type].shape
            new_node_parameters = np.zeros([shape[0], max_length], dtype=np.int)
            new_node_parameters[:, 0: shape[1]] = node_parameters[node_type]
            node_parameters[node_type] = new_node_parameters
            para_size_dict[node_type] = max_length
    else:
        para_size_dict = {i_node_type: 1 for i_node_type in node_type_allowed}

    return node_parameters, para_size_dict


def _collect_parameter_info(output_parameter,
                            parameter_type,
                            node_type,
                            default_dict,
                            info_dict):
    # step 1: get the parameter str
    if parameter_type in info_dict:
        # append the default setting into default_dict
        para_str = info_dict[parameter_type]
    elif parameter_type in default_dict[node_type]:
        para_str = default_dict[node_type][parameter_type]
    else:
        assert False, logger.error(
            'no information available for node: {}, para: {}'.format(
                node_type, parameter_type
            )
        )

    # step 2: parse the str into the parameter numbers
    if node_type == 'geom' and para_str in GEOM_TYPE_ENCODE:
        output_parameter.extend(GEOM_TYPE_ENCODE[para_str])
    else:
        output_parameter.extend(
            [float(element) for element in para_str.split(' ')]
        )

    return output_parameter


PARAMETERS_DEFAULT_DICT = {
    'root': {},
    'body': {'pos': 'NON_DEFAULT'},
    'geom': {
        'fromto': '-1 -1 -1 -1 -1 -1',
        'size': 'NON_DEFAULT',
        'type': 'NON_DEFAULT'
    },
    'joint': {
        'armature': '-1',
        'axis': 'NON_DEFAULT',
        'damping': '-1',
        'pos': 'NON_DEFAULT',
        'stiffness': '-1',
        'range': '-1 -1'
    }
}


def _get_para_list(xml_soup, node_type_allowed):
    '''
        @brief:
            for each type in the node_type_allowed, we find the attributes that
            shows up in the xml
            below is the node parameter info list:

            @root (size 0):
                More often the case, the root node is the domain root, as
                there is the 2d/3d information in it.

            @body (max size: 3):
                @pos: 3

            @geom (max size: 9):
                @fromto: 6
                @size: 1
                @type: 2

            @joint (max size: 11):
                @armature: 1
                @axis: 3
                @damping: 1
                @pos: 3
                @stiffness: 1  # important
                @range: 2
    '''
    # step 1: get the available parameter list for each node
    para_list = {node_type: [] for node_type in node_type_allowed}
    mj_soup = xml_soup.find('worldbody').find('body')
    for node_type in node_type_allowed:
        # search the node with type 'node_type'
        node_list = mj_soup.find_all(node_type)  # all the nodes
        for i_node in node_list:
            # deal with each node
            for key in i_node.attrs:
                # deal with each attributes
                if key not in para_list[node_type] and \
                        key in PARAMETERS_DEFAULT_DICT[node_type]:
                    para_list[node_type].append(key)

    # step 2: get default parameter settings
    default_dict = PARAMETERS_DEFAULT_DICT
    default_soup = xml_soup.find('default')
    if default_soup is not None:
        for node_type, para_type_list in para_list.iteritems():
            # find the default str if possible
            type_soup = default_soup.find(node_type)
            if type_soup is not None:
                for para_type in para_type_list:
                    if para_type in type_soup.attrs:
                        default_dict[node_type][para_type] = \
                            type_soup[para_type]
            else:
                logger.info(
                    'No default settings available for type {}'.format(
                        node_type
                    )
                )
    else:
        logger.warning('No default settings available for this xml!')

    return para_list, default_dict


if __name__ == '__main__':
    assert False
