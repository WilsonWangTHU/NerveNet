# ------------------------------------------------------------------------------
#   @brief:
#       The optimization agent is responsible for doing the updates.
#   @author:
#       modified from the code of kvfran, modified by Tingwu Wang
# ------------------------------------------------------------------------------


import numpy as np
import tensorflow as tf
import init_path
from util import utils
from network.baseline_network import tf_baseline_network
from network.gated_graph_baseline_network import tf_ggnn_baseline_network
from util import logger
from util import model_saver
from util import parallel_util
import os
from util import summary_handler
from agent import base_agent
from graph_util import graph_data_util


class optimization_agent(base_agent):
    '''
        @functions:
            @necessary components:
                def __init__(self, args, observation_size, action_size, task_q,
                             result_q, name_scope='trpo_agent')
                gef run(self)

            @build models:
                def build_models(self)
                def build_baseline_network(self)

            @build update op:
                def build_ppo_update_op(self)
                def build_trpo_update_op(self)

            @train the networks:
                def update_ppo_parameters(self, paths)
                def update_trpo_parameters(self, paths)

            @summary and ckpt:
                def record_summary_and_ckpt(self,
                                            paths, stats, ob_normalizer_info)
                def save_all(self, ob_normalizer_info={})
                def restore_all(self)

            @util function:
                def prepared_network_feeddict(self, data_dict)
                def generate_advantage(self, data_dict)
                def construct_minibatchFeeddict_from_feeddict(
                    self, feed_dict, minibatch_id_candidate,
                    current_id, batch_size
                )
                def prepare_feed_dict_map(self)
    '''

    def __init__(self,
                 args,
                 observation_size,
                 action_size,
                 task_q,
                 result_q,
                 name_scope='trpo_agent'):
        # the base agent
        super(optimization_agent, self).__init__(
            args=args,
            observation_size=observation_size,
            action_size=action_size,
            task_q=task_q,
            result_q=result_q,
            name_scope=name_scope
        )

        # the variables and networks to be used, init them before use them
        self.baseline_network = None
        self.env_info = None

        # used to save the checkpoint files
        self.best_reward = -np.inf
        self.last_save_iteration = 0
        self.timesteps_so_far = 0
        self._npr = np.random.RandomState(args.seed)

    def run(self):
        '''
            @brief:
                this is the standard function to be called by the
                "multiprocessing.Process"

            @NOTE:
                check the parallel_util.py for definitions
        '''
        self.build_models()

        # load the model if needed
        if self.args.ckpt_name is not None:
            self.restore_all()

        # the main training process
        while True:
            next_task = self.task_q.get()

            # Kill the learner
            if next_task is None or next_task == parallel_util.END_SIGNAL:
                self.task_q.task_done()
                break

            # Get the policy network weights
            elif next_task == parallel_util.START_SIGNAL:
                # just get the params of the network, no learning process
                self.task_q.task_done()
                self.result_q.put(self.get_policy())

            # Updating the network
            else:
                if self.args.test:
                    paths = next_task

                    paths.pop()
                    episoderewards = np.array(
                        [path["rewards"].sum() for path in paths]
                    )
                    self.task_q.task_done()
                    stats = {
                        "avg_reward": episoderewards.mean()
                    }
                    logger.info(stats)
                    return_data = {
                        'policy_weights': self.get_policy(),
                        'stats': stats,
                        'totalsteps': self.args.max_timesteps + 100,
                        'iteration': self.get_iteration_count(),

                        'std_reward': episoderewards.std(),
                        "avg_reward": episoderewards.mean(),
                        "max_reward": np.amax(episoderewards),
                        "min_reward": np.amin(episoderewards),
                        "median_reward": np.median(episoderewards),
                    }
                    self.result_q.put(return_data)

                # the actual training step
                else:
                    paths = next_task
                    stats = self.update_parameters(paths)
                    self.task_q.task_done()
                    return_data = {
                        'policy_weights': self.get_policy(),
                        'stats': stats,
                        'totalsteps': self.timesteps_so_far,
                        'iteration': self.get_iteration_count()
                    }
                    self.result_q.put(return_data)

    def build_models(self):
        '''
            @brief:
                this is the function where the rollout agents and optimization
                agent build their networks, set up the placeholders, and gather
                the variable list.
        '''
        # make sure that the agent has a session
        self.build_session()

        # set the summary writer
        self.summary_writer = summary_handler.gym_summary_handler(
            self.session, self.get_experiment_name(),
            enable=self.args.write_summary, summary_dir=self.args.output_dir
        )

        # build the policy network and baseline network
        self.build_policy_network()

        # the baseline function to reduce the variance
        self.build_baseline_network()

        # the training op and graphs
        self.build_ppo_update_op()
        self.update_parameters = self.update_ppo_parameters

        # init the network parameters (xavier initializer)
        self.session.run(tf.global_variables_initializer())

        # the set weight policy ops
        self.get_policy = utils.GetPolicyWeights(self.session,
                                                 self.policy_var_list)

        # prepare the feed_dict info for the ppo minibatches
        self.prepare_feed_dict_map()

        # prepared the init kl divergence if needed
        self.current_kl_lambda = 1

    def build_baseline_network(self):
        '''
            @brief:
                Build the baseline network, and fetch the baseline variable list
        '''

        # step 1: build the baseline network
        if self.args.use_gnn_as_policy:
            if self.args.use_gnn_as_value:
                self.baseline_network = tf_ggnn_baseline_network(
                    session=self.session,
                    name_scope=self.name_scope + '_baseline',
                    input_size=self.observation_size,
                    placeholder_list=self.gnn_placeholder_list,
                    ob_placeholder=None,
                    trainable=True,
                    build_network_now=True,
                    args=self.args
                )
            else:
                self.baseline_network = tf_baseline_network(
                    session=self.session,
                    name_scope=self.name_scope + '_baseline',
                    input_size=self.observation_size,
                    ob_placeholder=None,
                    trainable=True,
                    args=self.args
                )
                # in this case the raw obs and ob is different
                self.raw_obs_placeholder = \
                    self.baseline_network.get_ob_placeholder()
        else:
            self.baseline_network = tf_baseline_network(
                session=self.session,
                name_scope=self.name_scope + '_baseline',
                input_size=self.observation_size,
                ob_placeholder=self.obs_placeholder,
                args=self.args
            )
            assert not self.args.shared_network, logger.error(
                'shared fc policy not implemented'
            )
        # step 2: get the placeholders for the network
        self.target_return_placeholder = \
            self.baseline_network.get_target_return_placeholder()

    def build_ppo_update_op(self):
        '''
            @brief:
                The only difference from the vpg update is that here we clip the
                ratio
        '''
        self.build_update_op_preprocess()

        # step 1: the surrogate loss
        self.ratio_clip = tf.clip_by_value(self.ratio,
                                           1.0 - self.args.ppo_clip,
                                           1.0 + self.args.ppo_clip)

        # the pessimistic surrogate loss
        self.surr = tf.minimum(self.ratio_clip * self.advantage_placeholder,
                               self.ratio * self.advantage_placeholder)
        self.surr = -tf.reduce_mean(self.surr)

        # step 2: the value function loss. if not tf, the loss will be fit in
        # update_parameters_postprocess
        self.vf_loss = self.baseline_network.get_vf_loss()
        self.loss = self.surr

        # step 3: the kl penalty term
        if self.args.use_kl_penalty:
            self.loss += self.kl_lambda_placeholder * self.kl
            self.loss += self.args.kl_eta * \
                tf.square(tf.maximum(0.0, self.kl - 2.0 * self.args.target_kl))

        # step 4: weight decay
        self.weight_decay_loss = 0.0
        for var in tf.trainable_variables():
            self.weight_decay_loss += tf.nn.l2_loss(var)
        if self.args.use_weight_decay:
            self.loss += self.weight_decay_loss * self.args.weight_decay_coeff

        # step 5: build the optimizer
        self.lr_placeholder = tf.placeholder(tf.float32, [],
                                             name='learning_rate')
        self.current_lr = self.args.lr
        if self.args.use_gnn_as_policy:
            # we need to clip the gradient for the ggnn
            self.optimizer = tf.train.AdamOptimizer(self.lr_placeholder)
            self.tvars = tf.trainable_variables()
            self.grads = tf.gradients(self.loss, self.tvars)
            self.clipped_grads, _ = tf.clip_by_global_norm(
                self.grads,
                self.args.grad_clip_value,
                name='clipping_gradient'
            )

            self.update_op = self.optimizer.apply_gradients(
                zip(self.clipped_grads, self.tvars)
            )
        else:
            self.update_op = tf.train.AdamOptimizer(
                learning_rate=self.lr_placeholder, epsilon=1e-5
            ).minimize(self.loss)

            self.tvars = tf.trainable_variables()
            self.grads = tf.gradients(self.loss, self.tvars)

        if self.args.shared_network:
            self.update_vf_op = tf.train.AdamOptimizer(
                learning_rate=self.lr_placeholder, epsilon=1e-5
            ).minimize(self.vf_loss)
        else:
            self.update_vf_op = tf.train.AdamOptimizer(
                learning_rate=self.args.value_lr, epsilon=1e-5
            ).minimize(self.vf_loss)

    def update_ppo_parameters(self, paths):
        '''
            @brief: update the ppo
        '''
        # step 1: get the data dict
        ob_normalizer_info = paths.pop()
        feed_dict = self.prepared_network_feeddict(paths)

        # step 2: train the network
        logger.info('| %11s | %11s | %11s | %11s| %11s|' %
                    ('surr', 'kl', 'ent', 'vf_loss', 'weight_l2'))
        self.timesteps_so_far += self.args.timesteps_per_batch
        for i_epochs in range(self.args.optim_epochs +
                              self.args.extra_vf_optim_epochs):

            minibatch_id_candidate = range(
                feed_dict[self.action_placeholder].shape[0]
            )
            self._npr.shuffle(minibatch_id_candidate)
            # make sure that only timesteps per batch is used
            minibatch_id_candidate = \
                minibatch_id_candidate[: self.args.timesteps_per_batch]
            current_id = 0

            surrogate_epoch, kl_epoch, entropy_epoch, vf_epoch, weight_epoch = \
                [], [], [], [], []
            while current_id + self.args.optim_batch_size <= \
                    len(minibatch_id_candidate) and current_id >= 0:

                # fetch the minidata batch
                sub_feed_dict, current_id, minibatch_id_candidate = \
                    self.construct_minibatchFeeddict_from_feeddict(
                        feed_dict, minibatch_id_candidate, current_id,
                        self.args.optim_batch_size,
                        is_all_feed=self.args.minibatch_all_feed
                    )

                if i_epochs < self.args.optim_epochs:
                    # train for one iteration in this epoch
                    _, i_surrogate_mini, i_kl_mini, i_entropy_mini, \
                        i_weight_mini = self.session.run(
                            [self.update_op, self.surr, self.kl, self.ent,
                                self.weight_decay_loss],
                            feed_dict=sub_feed_dict
                        )

                    # train the value network with fixed network coeff
                    _, i_vf_mini = self.session.run(
                        [self.update_vf_op, self.vf_loss],
                        feed_dict=sub_feed_dict
                    )
                    surrogate_epoch.append(i_surrogate_mini)
                    kl_epoch.append(i_kl_mini)
                    entropy_epoch.append(i_entropy_mini)
                    vf_epoch.append(i_vf_mini)
                    weight_epoch.append(i_weight_mini)
                else:
                    # only train the value function, might be unstable if share
                    # the value network and policy network
                    _, i_vf_mini = self.session.run(
                        [self.update_vf_op, self.vf_loss],
                        feed_dict=sub_feed_dict
                    )
                    vf_epoch.append(i_vf_mini)

            if i_epochs < self.args.optim_epochs:
                surrogate_epoch = np.mean(surrogate_epoch)
                kl_epoch = np.mean(kl_epoch)
                entropy_epoch = np.mean(entropy_epoch)
                vf_epoch = np.mean(vf_epoch)
                weight_epoch = np.mean(weight_epoch)
            else:
                surrogate_epoch = -0.1
                kl_epoch = -0.1
                entropy_epoch = -0.1
                weight_epoch = -0.1
                vf_epoch = np.mean(vf_epoch)

            # if we use kl_penalty, we will do early stopping if needed
            if self.args.use_kl_penalty:
                assert self.args.minibatch_all_feed, logger.error(
                    'KL penalty not available for epoch minibatch training'
                )
                if kl_epoch > 4 * self.args.target_kl and \
                        self.args.minibatch_all_feed:
                    logger.info('Early Stopping')
                    break

            logger.info(
                '| %10.8f | %10.8f | %10.4f | %10.4f | %10.4f |' %
                (surrogate_epoch, kl_epoch, entropy_epoch,
                 vf_epoch, weight_epoch)
            )

        i_surrogate_total, i_kl_total, i_entropy_total, \
            i_vf_total, i_weight_total = self.session.run(
                [self.surr, self.kl, self.ent,
                    self.vf_loss, self.weight_decay_loss],
                feed_dict=feed_dict
            )

        # step 3: update the hyperparameters of updating
        self.update_adaptive_hyperparams(kl_epoch, i_kl_total)

        # step 4: record the stats
        stats = {}

        episoderewards = np.array(
            [path["rewards"].sum() for path in paths]
        )
        stats["avg_reward"] = episoderewards.mean()
        stats["entropy"] = i_entropy_total
        stats["kl"] = i_kl_total
        stats["surr_loss"] = i_surrogate_total
        stats["vf_loss"] = i_vf_total
        stats["weight_l2_loss"] = i_weight_total
        stats['learning_rate'] = self.current_lr

        if self.args.use_kl_penalty:
            stats['kl_lambda'] = self.current_kl_lambda

        # step 5: record the summary and save checkpoints
        self.record_summary_and_ckpt(paths, stats, ob_normalizer_info)

        return stats

    def update_adaptive_hyperparams(self, kl_epoch, i_kl_total):
        # update the lambda of kl divergence
        if self.args.use_kl_penalty:
            if kl_epoch > self.args.target_kl_high * self.args.target_kl:
                self.current_kl_lambda *= self.args.kl_alpha
                if self.current_kl_lambda > 30 and \
                        self.current_lr > 0.1 * self.args.lr:
                    self.current_lr /= 1.5
            elif kl_epoch < self.args.target_kl_low * self.args.target_kl:
                self.current_kl_lambda /= self.args.kl_alpha
                if self.current_kl_lambda < 1 / 30 and \
                        self.current_lr < 10 * self.args.lr:
                    self.current_lr *= 1.5

            self.current_kl_lambda = max(self.current_kl_lambda, 1 / 35.0)
            self.current_kl_lambda = min(self.current_kl_lambda, 35.0)

        # update the lr
        elif self.args.lr_schedule == 'adaptive':
            mean_kl = i_kl_total
            if mean_kl > self.args.target_kl_high * self.args.target_kl:
                self.current_lr /= self.args.lr_alpha
            if mean_kl < self.args.target_kl_low * self.args.target_kl:
                self.current_lr *= self.args.kl_alpha

            self.current_lr = max(self.current_lr, 3e-10)
            self.current_lr = min(self.current_lr, 1e-2)
        else:
            self.current_lr = self.args.lr * max(
                1.0 - float(self.timesteps_so_far) / self.args.max_timesteps,
                0.0
            )

    def record_summary_and_ckpt(self, paths, stats, ob_normalizer_info):
        # logger the information and write summary
        for k, v in stats.iteritems():
            logger.info(k + ": " + " " * (40 - len(k)) + str(v))

        current_iteration = self.get_iteration_count()

        if current_iteration % self.args.min_ckpt_iteration_diff == 0:
            logger.info('------------- Printing hyper-parameters -----------')
            for key, val in self.args.__dict__.iteritems():
                logger.info('{}: {}'.format(key, val))
            logger.info('experiment name: '.format(self.get_experiment_name()))

        # add the summary
        if current_iteration % self.args.summary_freq == 0:
            for key, val in stats.items():
                self.summary_writer.manually_add_scalar_summary(
                    key, val, current_iteration
                )

        # save the model if needed
        current_reward = stats["avg_reward"]
        if self.best_reward < current_reward:
            if current_iteration > self.args.checkpoint_start_iteration and \
                    current_iteration - self.last_save_iteration > \
                    self.args.min_ckpt_iteration_diff:
                self.best_reward = current_reward

                # save the model
                self.save_all(current_iteration, ob_normalizer_info)

                self.last_save_iteration = current_iteration

        if self.best_reward > -np.inf:  # it means that we updated it
            logger.info('Current max reward: {}'.format(self.best_reward))

        # add the iteration count
        self.session.run(self.iteration_add_op)

    def save_all(self, current_iteration, ob_normalizer_info={}):
        '''
            @brief: save the model into several npy file.
        '''
        model_name = self.get_output_path(save=True)
        # save multiple models:
        model_name += '_' + str(current_iteration)

        # save the parameters of the policy network, baseline network
        # and the ob_normalizer
        self.policy_network.save_checkpoint(model_name + '_policy.npy')
        self.baseline_network.save_checkpoint(model_name + '_baseline.npy')
        model_saver.save_numpy_model(model_name + '_normalizer.npy',
                                     ob_normalizer_info)

        logger.info(
            '[SAVE_CKPT] saving files under the name of {}'.format(model_name)
        )

    def restore_all(self):
        '''
            @brief: restore the parameters
        '''
        model_name = self.get_output_path(save=False)

        # load the parameters one by one
        self.policy_network.load_checkpoint(
            model_name + '_policy.npy',
            transfer_env=self.args.transfer_env,
            logstd_option=self.args.logstd_option,
            gnn_option_list=[self.args.gnn_node_option,
                             self.args.root_connection_option,
                             self.args.gnn_output_option,
                             self.args.gnn_embedding_option],
            mlp_raw_transfer=self.args.mlp_raw_transfer
        )
        self.baseline_network.load_checkpoint(model_name + '_baseline.npy')
        self.last_save_iteration = self.get_iteration_count()

        logger.info(
            '[LOAD_CKPT] saving files under the name of {}'.format(
                model_name
            )
        )
        logger.info(
            '[LOAD_CKPT] The normalizer is loaded in the rollout agents'
        )

    def get_output_path(self, save=True):
        if save:
            if self.args.output_dir is None:
                path = init_path.get_base_dir()
                path = os.path.abspath(path)
            else:
                path = os.path.abspath(self.args.output_dir)
            base_path = os.path.join(path, 'checkpoint')
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            model_name = os.path.join(base_path, self.get_experiment_name())
        else:
            path = self.args.ckpt_name
            model_name = path
        return model_name

    def generate_advantage(self, data_dict, feed_dict):
        '''
            @brief: calculate the parameters for the advantage function
        '''
        # get the baseline function
        if self.args.use_gnn_as_value:
            baseline_data = self.baseline_network.predict(feed_dict)
            current_id = 0
            for path in data_dict:
                path['baseline'] = baseline_data[
                    current_id: current_id + len(path['rewards'])
                ]
                current_id += len(path['rewards'])

            assert current_id == len(baseline_data), logger.error(
                'Extra baseline predicted? ({} vs {})'.format(
                    current_id, len(baseline_data)
                )
            )
        else:
            for path in data_dict:
                # the predicted value function (baseline function)
                path["baseline"] = self.baseline_network.predict(path)

        # esitmate the advantages
        if self.args.advantage_method == 'raw':
            for path in data_dict:
                # the gamma discounted rollout value function
                path["returns"] = utils.discount(path["rewards"],
                                                 self.args.gamma)
                path["advantage"] = path["returns"] - path["baseline"]
                path['target_return'] = path['returns']
        else:
            assert self.args.advantage_method == 'gae', logger.error(
                'invalid advantage estimation method: {}'.format(
                    self.args.advantage_method
                )
            )

            for path in data_dict:
                # the gamma discounted rollout value function
                path["returns"] = utils.discount(path["rewards"],
                                                 self.args.gamma)

                # init the advantage
                path["advantage"] = np.zeros(path['returns'].shape)

                num_steps = len(path['returns'])

                # generate the GAE advantage
                for i_step in reversed(range(num_steps)):
                    if i_step < num_steps - 1:
                        delta = path['rewards'][i_step] \
                            + self.args.gamma * path['baseline'][i_step + 1] \
                            - path['baseline'][i_step]
                        path['advantage'][i_step] = \
                            delta + self.args.gamma * self.args.gae_lam \
                            * path['advantage'][i_step + 1]
                    else:
                        delta = path['rewards'][i_step] \
                            - path['baseline'][i_step]
                        path['advantage'][i_step] = delta

                path['target_return'] = path['advantage'] + path['baseline']

        # standardized advantage function
        advant_n = np.concatenate([path["advantage"] for path in data_dict])
        advant_n -= advant_n.mean()
        advant_n /= (advant_n.std() + 1e-8)  # standardize to mean 0 stddev 1
        return advant_n

    def prepared_network_feeddict(self, data_dict):
        '''
            @brief:
                For the graph neural network, we will need to change the size
                from [batch_size, ob_dim] into [batch_size * node_size, ob_dim].
            @return:
                @1. feed_dict for trpo or ppo update
                    @self.action_placeholder
                    @self.advantage_placeholder
                    @self.oldaction_dist_mu_placeholder
                    @self.oldaction_dist_logstd_placeholder
                    @self.batch_size_float_placeholder

                    # baseline function
                    @self.target_return_placeholder (for ppo only)

                @2. feed_dict for generating the policy action
                    @self.obs_placeholder

                    # for ggnn only
                    @self.graph_obs_placeholder
                    @self.graph_parameters_placeholder
                    @self.batch_size_int_placeholder
                    @self.receive_idx_placeholder
                    @self.send_idx_placeholder
                    @self.inverse_node_type_idx_placeholder
                    @self.output_idx_placeholder

                @3. feed_dict for baseline if baseline is a fc-policy and policy is
                    a ggnn policy
                    @self.raw_obs_placeholder

                NOTE: for minibatch training, check @prepare_feed_dict_map and
                    @construct_minibatchFeeddict_from_feeddict
        '''
        feed_dict = {}
        # step 1: For ggnn and fc policy, we have different obs in feed_dict
        obs_n = np.concatenate([path["obs"] for path in data_dict])
        feed_dict.update(self.prepared_policy_network_feeddict(obs_n))

        # step 2: prepare the advantage function, old action / obs needed for
        # trpo / ppo updates
        advant_n = self.generate_advantage(data_dict, feed_dict)
        action_dist_mu = np.concatenate(
            [path["action_dists_mu"] for path in data_dict]
        )
        action_dist_logstd = np.concatenate(
            [path["action_dists_logstd"] for path in data_dict]
        )
        action_n = np.concatenate([path["actions"] for path in data_dict])
        feed_dict.update(
            {self.action_placeholder: action_n,
             self.advantage_placeholder: advant_n,
             self.oldaction_dist_mu_placeholder: action_dist_mu,
             self.oldaction_dist_logstd_placeholder: action_dist_logstd,
             self.batch_size_float_placeholder: np.array(float(len(obs_n)))}
        )

        # step 3: feed_dict to update value function
        target_return = \
            np.concatenate([path["target_return"]
                            for path in data_dict])
        feed_dict.update(
            {self.target_return_placeholder: target_return}
        )

        # step 4: the kl penalty and learning rate
        if self.args.use_kl_penalty:
            feed_dict.update(
                {self.kl_lambda_placeholder: self.current_kl_lambda}
            )
        feed_dict.update({self.lr_placeholder: self.current_lr})

        return feed_dict

    def construct_minibatchFeeddict_from_feeddict(self,
                                                  feed_dict,
                                                  minibatch_id_candidate,
                                                  current_id,
                                                  batch_size,
                                                  is_all_feed=False):
        '''
            @brief:
                construct minibatch feed_dict from the database. If we set
                minibatch_all_feed = 1, then we just feed the whole dataset
                into the feed_dict

            @elemenet to process:
                @self.batch_feed_dict_key:
                @self.graph_batch_feed_dict_key
                @self.static_feed_dict:
                @self.dynamical_feed_dict_key:
                @self.graph_index_feed_dict:
        '''
        sub_feed_dict = {}
        if not is_all_feed:

            # step 0: id validation and id picking
            assert batch_size <= len(minibatch_id_candidate), \
                logger.error('Please increse the rollout size!')
            if current_id + batch_size > len(minibatch_id_candidate):
                logger.warning('shuffling the ids')
                current_id = 0
                self._npr.shuffle(minibatch_id_candidate)
            candidate_id = minibatch_id_candidate[
                current_id: current_id + batch_size
            ]

            # step 1: gather sub feed dict by sampling from the feed_dict
            for key in self.batch_feed_dict_key:
                sub_feed_dict[key] = feed_dict[key][candidate_id]

            # step 2: update the graph_batch_feed_dict
            if self.args.use_gnn_as_policy:
                for node_type in self.node_info['node_type_dict']:
                    num_nodes = len(self.node_info['node_type_dict'][node_type])
                    graph_candidate_id = [
                        range(i_id * num_nodes, (i_id + 1) * num_nodes)
                        for i_id in candidate_id
                    ]

                    # flatten the ids
                    graph_candidate_id = sum(graph_candidate_id, [])
                    for key in self.graph_batch_feed_dict_key:
                        sub_feed_dict[key[node_type]] = \
                            feed_dict[key[node_type]][graph_candidate_id]

            # step 3: static elements which are invariant to batch_size
            sub_feed_dict.update(self.static_feed_dict)

            # step 4: update dynamical elements
            for key in self.dynamical_feed_dict_key:
                sub_feed_dict[key] = feed_dict[key]

            # step 6: get the graph index feed_dict
            if self.args.use_gnn_as_policy:
                sub_feed_dict.update(self.graph_index_feed_dict)

            # step 5: update current id
            current_id = current_id + batch_size  # update id
        else:
            # feed the whole feed_dictionary into the network
            sub_feed_dict = feed_dict
            current_id = -1  # -1 means invalid

        return sub_feed_dict, current_id, minibatch_id_candidate

    def prepare_feed_dict_map(self):
        '''
            @brief:
                When trying to get the sub diction in
                @construct_minibatchFeeddict_from_feeddict, some key are just
                directly transferable. While others might need some other work

                @1. feed_dict for trpo or ppo update

                    # baseline function

                @2. feed_dict for generating the policy action

                    # for ggnn only
                @3. feed_dict for baseline if baseline is a fc-policy and policy
                    is a ggnn policy

            @return:
                @self.batch_feed_dict_key:
                    Shared between the fc policy network and ggnn network.
                    Most of them are only used for the update.

                        @self.action_placeholder
                        @self.advantage_placeholder
                        @self.oldaction_dist_mu_placeholder
                        @self.oldaction_dist_logstd_placeholder

                        (if use fc policy)
                        @self.obs_placeholder

                        (if use_ggn and baseline not gnn)
                        @self.raw_obs_placeholder

                        (if use tf baseline)
                        @self.target_return_placeholder (for ppo only)

                @self.graph_batch_feed_dict_key
                    Used by the ggnn. This feed_dict key list is a little bit
                    different from @self.batch_feed_dict_key if we want to do
                    minibatch

                        @self.graph_obs_placeholder
                        @self.graph_parameters_placeholder

                @self.static_feed_dict:
                    static elements that are set by optim parameters. These
                    parameters are set differently between minibatch_all_feed
                    equals 0 / equals 1

                        @self.batch_size_float_placeholder
                        @self.batch_size_int_placeholder

                @self.dynamical_feed_dict_key:
                    elements that could be changing from time to time

                        @self.kl_lambda_placeholder
                        @self.lr_placeholder

                @self.graph_index_feed_dict:
                    static index for the ggnn.

                        @self.receive_idx_placeholder
                        @self.inverse_node_type_idx_placeholder
                        @self.output_idx_placeholder
                        @self.send_idx_placeholder[i_edge]
                        @self.node_type_idx_placeholder[i_node_type]

        '''
        # step 1: gather the key for batch_feed_dict
        self.batch_feed_dict_key = [
            self.action_placeholder,
            self.advantage_placeholder,
            self.oldaction_dist_mu_placeholder,
            self.oldaction_dist_logstd_placeholder
        ]

        if not self.args.use_gnn_as_policy:
            self.batch_feed_dict_key.append(self.obs_placeholder)

        if self.args.use_gnn_as_policy and not self.args.use_gnn_as_value:
            self.batch_feed_dict_key.append(self.raw_obs_placeholder)

        self.batch_feed_dict_key.append(self.target_return_placeholder)

        # step 2: gather the graph batch feed_dict
        self.graph_batch_feed_dict_key = []
        if self.args.use_gnn_as_policy:
            self.graph_batch_feed_dict_key.extend(
                [self.graph_obs_placeholder, self.graph_parameters_placeholder]
            )

        # step 2: gather the static feed_dictionary
        self.static_feed_dict = {
            self.batch_size_float_placeholder:
                np.array(float(self.args.optim_batch_size))
        }
        if self.args.use_gnn_as_policy:
            self.static_feed_dict.update(
                {self.batch_size_int_placeholder: self.args.optim_batch_size}
            )

        # step 3: gather the dynamical feed_dictionary
        self.dynamical_feed_dict_key = []
        if self.args.use_kl_penalty:
            self.dynamical_feed_dict_key.append(self.kl_lambda_placeholder)
        self.dynamical_feed_dict_key.append(self.lr_placeholder)

        # step 4: gather the graph_index feed_dict
        if self.args.use_gnn_as_policy:
            # construct a dummy obs to pass the batch size info
            dummy_obs = np.zeros([self.args.optim_batch_size, 10])
            # print dummy_obs.shape
            node_info = self.policy_network.get_node_info()

            # get the index for minibatches
            _, _, receive_idx, send_idx, \
                node_type_idx, inverse_node_type_idx, \
                output_type_idx, inverse_output_type_idx, _ = \
                graph_data_util.construct_graph_input_feeddict(
                    node_info,
                    dummy_obs, -1, -1, -1, -1, -1, -1, -1,
                    request_data=['idx']
                )

            self.graph_index_feed_dict = {
                self.receive_idx_placeholder:
                    receive_idx,
                self.inverse_node_type_idx_placeholder:
                    inverse_node_type_idx,
                self.inverse_output_type_idx_placeholder:
                    inverse_output_type_idx
            }

            # append the send idx
            for i_edge in node_info['edge_type_list']:
                self.graph_index_feed_dict[
                    self.send_idx_placeholder[i_edge]
                ] = send_idx[i_edge]

            # append the node type idx
            for i_node_type in node_info['node_type_dict']:
                self.graph_index_feed_dict[
                    self.node_type_idx_placeholder[i_node_type]
                ] = node_type_idx[i_node_type]

            # append the node type idx
            for i_output_type in node_info['output_type_dict']:
                self.graph_index_feed_dict[
                    self.output_type_idx_placeholder[i_output_type]
                ] = output_type_idx[i_output_type]
