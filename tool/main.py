# -----------------------------------------------------------------------------
#   @brief: main fucntion
# -----------------------------------------------------------------------------

import gym
import init_path
from agent import optimization_agent
from agent import rollout_master_agent
from config.config import get_config
from util import parallel_util
from util import logger
import multiprocessing
import time
from environments import register
init_path.bypass_frost_warning()

if __name__ == '__main__':
    # get the configuration
    logger.info('New environments available : {}'.format(
        register.get_name_list()))
    args = get_config()
    args.max_pathlength = gym.spec(args.task).timestep_limit
    learner_env = gym.make(args.task)

    if args.write_log:
        logger.set_file_handler(
            path=args.output_dir,
            prefix='mujoco_' + args.task, time_str=args.time_id
        )

    learner_tasks = multiprocessing.JoinableQueue()
    learner_results = multiprocessing.Queue()
    learner_agent = optimization_agent.optimization_agent(
        args,
        learner_env.observation_space.shape[0],
        learner_env.action_space.shape[0],
        learner_tasks,
        learner_results
    )
    learner_agent.start()

    # the rollouts agents
    rollout_agent = rollout_master_agent.parallel_rollout_master_agent(
        args,
        learner_env.observation_space.shape[0],
        learner_env.action_space.shape[0]
    )

    # start the training and rollouting process
    learner_tasks.put(parallel_util.START_SIGNAL)
    learner_tasks.join()
    starting_weights = learner_results.get()
    rollout_agent.set_policy_weights(starting_weights)

    # some training stats
    start_time = time.time()
    totalsteps = 0

    while True:

        # runs a bunch of async processes that collect rollouts
        rollout_start = time.time()
        paths = rollout_agent.rollout()
        rollout_time = (time.time() - rollout_start) / 60.0

        learn_start = time.time()
        learner_tasks.put(paths)
        learner_tasks.join()
        results = learner_results.get()
        totalsteps = results['totalsteps']
        learn_time = (time.time() - learn_start) / 60.0

        # update the policy
        rollout_agent.set_policy_weights(results['policy_weights'])

        logger.info(
            "------------- Iteration %d --------------" % results['iteration']
        )
        logger.info(
            "total time: %.2f mins" % ((time.time() - start_time) / 60.0)
        )
        logger.info("optimization agent spent : %.2f mins" % (learn_time))
        logger.info("rollout agent spent : %.2f mins" % (rollout_time))

        logger.info("%d total steps have happened" % totalsteps)

        if totalsteps > args.max_timesteps:
            break

    rollout_agent.end()
    learner_tasks.put(parallel_util.END_SIGNAL)  # kill the learner
    if args.test:
        logger.info(
            'Test performance ({} rollouts): {}'.format(
                args.test, results['avg_reward']
            )
        )

        logger.info(
            'max: {}, min: {}, median: {}'.format(
                results['max_reward'], results['min_reward'],
                results['median_reward']
            )
        )
