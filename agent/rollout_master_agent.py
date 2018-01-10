# -----------------------------------------------------------------------------
#   @brief:
#       modified from the code of kvfran
#   @modified:
#       Tingwu Wang
# -----------------------------------------------------------------------------


import multiprocessing
import init_path
import os
from util import logger
from util import parallel_util
from util import model_saver
from six.moves import xrange
from rollout_agent import rollout_agent
import numpy as np
from graph_util import structure_mapper


class parallel_rollout_master_agent():

    def __init__(self, args, observation_size, action_size):
        '''
            @brief:
                the master agent has several actors (or samplers) to do the
                sampling for it.
        '''
        self.args = args
        self.observation_size = observation_size
        self.action_size = action_size

        # init the running means
        self.running_mean_info = {
            'mean': 0.0,
            'variance': 1,
            'step': 0.01,
            'square_sum': 0.01,
            'sum': 0.0
        }
        self.load_running_means()

        # init the multiprocess actors
        self.tasks = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        self.init_actors()

        # we will start by running 20,000 / 1000 = 20 episodes for the first
        # iteration
        self.average_timesteps_in_episode = args.max_pathlength

    def rollout(self):
        # step 1: ask the samplers to generate rollouts
        rollout_data = self.ask_for_rollouts()

        # step 2: update the running means
        self.update_running_means(rollout_data)

        # step 3: synchronize the filter statistics for all samplers
        for i_agent in xrange(self.args.num_threads + 1):
            self.tasks.put(
                (parallel_util.AGENT_SYNCHRONIZE_FILTER, self.running_mean_info)
            )
        self.tasks.join()

        # tell the trpo agent about the information of normalizer
        rollout_data.append(self.running_mean_info)
        return rollout_data

    def set_policy_weights(self, parameters):
        for i_agent in xrange(self.args.num_threads + 1):
            self.tasks.put((parallel_util.AGENT_SET_POLICY_WEIGHTS, parameters))
        self.tasks.join()

    def end(self):
        for i_agent in xrange(self.args.num_threads + 1):
            self.tasks.put((parallel_util.END_ROLLOUT_SIGNAL, None))

    def ask_for_rollouts(self):
        '''
            @brief:
                Run the experiments until a total of @timesteps_per_batch
                timesteps are collected.
        '''
        num_timesteps_received = 0
        timesteps_needed = self.args.timesteps_per_batch
        rollout_data = []

        while True:
            # calculate the expected number of episodes needed
            num_rollouts = int(
                np.ceil(timesteps_needed / self.average_timesteps_in_episode)
            )

            if self.args.test:
                num_rollouts = self.args.test

            # request episodes
            for i_agent in xrange(num_rollouts):
                self.tasks.put((parallel_util.START_SIGNAL, None))
            self.tasks.join()

            # collect episodes
            for _ in range(num_rollouts):
                traj_episode = self.results.get()
                rollout_data.append(traj_episode)
                num_timesteps_received += len(traj_episode['rewards'])

            # update average timesteps per episode
            self.average_timesteps_in_episode = \
                float(num_timesteps_received) / len(rollout_data)

            timesteps_needed = self.args.timesteps_per_batch - \
                num_timesteps_received

            if timesteps_needed <= 0 or self.args.test:
                break

        logger.info('Rollouts generating ...')
        logger.info('{} time steps from {} episodes collected'.format(
            num_timesteps_received, len(rollout_data)
        ))

        return rollout_data

    def update_running_means(self, paths):
        # collect the info
        new_sum = 0.0
        new_step_sum = 0.0
        new_sq_sum = 0.0
        for path in paths:
            raw_obs = path['raw_obs']
            new_sum += raw_obs.sum(axis=0)
            new_step_sum += raw_obs.shape[0]
            new_sq_sum += (np.square(raw_obs)).sum(axis=0)

        # update the parameters
        self.running_mean_info['sum'] += new_sum
        self.running_mean_info['square_sum'] += new_sq_sum
        self.running_mean_info['step'] += new_step_sum
        self.running_mean_info['mean'] = \
            self.running_mean_info['sum'] / self.running_mean_info['step']

        self.running_mean_info['variance'] = np.maximum(
            self.running_mean_info['square_sum'] /
            self.running_mean_info['step'] -
            np.square(self.running_mean_info['mean']),
            1e-2
        )

    def load_running_means(self):
        # load the observation running mean
        if self.args.ckpt_name is not None:
            base_path = os.path.join(init_path.get_base_dir(), 'checkpoint')
            logger.info('[LOAD_CKPT] loading observation normalizer info')
            self.running_mean_info = model_saver.load_numpy_model(
                os.path.join(base_path,
                             self.args.ckpt_name + '_normalizer.npy'),
                numpy_var_list=self.running_mean_info
            )
            self.running_mean_info['transfer_env'] = self.args.transfer_env

        if not self.args.transfer_env == 'Nothing2Nothing':
            if self.args.mlp_raw_transfer == 0:
                assert 'shared' in self.args.gnn_embedding_option

            ienv, oenv = [env + '-v1'
                          for env in self.args.transfer_env.split('2')]
            self.running_mean_info = \
                structure_mapper.map_transfer_env_running_mean(
                    ienv, oenv, self.running_mean_info,
                    self.observation_size,
                    self.args.gnn_node_option, self.args.root_connection_option,
                    self.args.gnn_output_option, self.args.gnn_embedding_option
                )

    def init_actors(self):
        '''
            @brief: init the actors and start the multiprocessing
        '''

        self.actors = []
        # the main actor who can sampler video result
        self.actors.append(
            rollout_agent(self.args,
                          self.observation_size,
                          self.action_size,
                          self.tasks,
                          self.results,
                          0,
                          self.args.monitor,
                          init_filter_parameters=self.running_mean_info)
        )

        # the sub actor that only do the sampling
        for i in xrange(self.args.num_threads):
            self.actors.append(
                rollout_agent(
                    self.args,
                    self.observation_size,
                    self.action_size,
                    self.tasks,
                    self.results,
                    i + 1,
                    False,
                    init_filter_parameters=self.running_mean_info
                )
            )

        for i_actor in self.actors:
            i_actor.start()
