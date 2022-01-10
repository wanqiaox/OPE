# Imports
import numpy as np
import json
from copy import deepcopy

from ope.envs.graph import Graph
from ope.policies.basics import BasicPolicy

from ope.experiment_tools.experiment import ExperimentRunner, analysis
from ope.experiment_tools.config import Config
from ope.experiment_tools.factory import setup_params

# Get configuration
configuration_filename = "toy_graph_pomdp_cfg.json"
with open('COBS-master/cfgs/{0}'.format(configuration_filename), 'r') as f:
    param = json.load(f)

param = setup_params(param) # Setup parameters
runner = ExperimentRunner() # Instantiate a runner for an experiment

# run 5 experiments, each with a varying number of trajectories
for N in range(5):
    
    configuration = deepcopy(param['experiment']) # Make sure to deepcopy as to never change original
    #configuration['num_traj'] = 10000
    configuration['num_traj'] = 8*2**N # Increase dataset size

    # store these credentials in an object
    cfg = Config(configuration)

    # initialize environment with the parameters from the config file.
    # If you'd like to use a different environment, swap this line
    env = Graph(make_pomdp=cfg.is_pomdp,
                number_of_pomdp_states=cfg.pomdp_horizon,
                transitions_deterministic=not cfg.stochastic_env,
                max_length=cfg.horizon,
                sparse_rewards=cfg.sparse_rewards,
                stochastic_rewards=cfg.stochastic_rewards)

    # set seed for the experiment
    np.random.seed(cfg.seed)

    # processor processes the state for storage,  {(processor(x), a, r, processor(x'), done)}
    processor = lambda x: x

    # absorbing state for padding if episode ends before horizon is reached. This is environment dependent.
    absorbing_state = processor(np.array([env.n_dim - 1]))

    # Setup policies. BasicPolicy takes the form [P(a=0), P(a=1), ..., P(a=n)]
    # For different policies, swap in here
    actions = [0, 1]
    pi_e = BasicPolicy(
        actions, [max(.001, cfg.eval_policy), 1 - max(.001, cfg.eval_policy)])
    pi_b = BasicPolicy(
        actions, [max(.001, cfg.base_policy), 1 - max(.001, cfg.base_policy)])

    # add env, policies, absorbing state and processor
    cfg.add({
        'env': env,
        'pi_e': pi_e,
        'pi_b': pi_b,
        'processor': processor,
        'absorbing_state': absorbing_state
    })
    cfg.add({'models': param['models']})

    # Add the configuration
    runner.add(cfg)

# Run the experiments
results = runner.run()

# Analyze the results
# Each row in the result is (OPE estimator, V(pi_e), MSE Error from on-policy: (V(pi_e) - True)**2)
for num, result in enumerate(results):
    print('Result Experiment %s' % (num+1))
    analysis(result)
    print('*'*20)
    print()