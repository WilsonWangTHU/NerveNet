# -----------------------------------------------------------------------------
#   @brief:
#       Define some signals used during parallel
# -----------------------------------------------------------------------------

# it makes the main trpo agent push its weights into the tunnel
START_SIGNAL = 1

# it ends the training
END_SIGNAL = -1

# ends the rollout
END_ROLLOUT_SIGNAL = -2

# ask the rollout agents to collect the ob normalizer's info
AGENT_COLLECT_FILTER_INFO = 10

# ask the rollout agents to synchronize the ob normalizer's info
AGENT_SYNCHRONIZE_FILTER = 20

# ask the agents to set their parameters of network
AGENT_SET_POLICY_WEIGHTS = 30
