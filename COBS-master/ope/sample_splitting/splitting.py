import numpy as np
from copy import deepcopy

from ope.algos.doubly_robust_v2 import DoublyRobust_v2 as DR
from ope.algos.fqe import FittedQEvaluation
from ope.algos.magic import MAGIC
from ope.algos.average_model import AverageModel as AM
from ope.algos.sequential_DR import SeqDoublyRobust as SeqDR
from ope.algos.dm_regression import DirectMethodRegression as DM
from ope.algos.traditional_is import TraditionalIS as IS
from ope.algos.infinite_horizon import InfiniteHorizonOPE as IH
from ope.algos.dm_regression import DirectMethodRegression
from ope.algos.more_robust_doubly_robust import MRDR
from ope.algos.retrace_lambda import Retrace
from ope.algos.approximate_model import ApproxModel

from ope.policies.basics import BasicPolicy
from ope.policies.epsilon_greedy_policy import EGreedyPolicy
from ope.policies.tabular_model import TabularPolicy

from ope.utls.rollout import rollout
from ope.utls.rollout import Data

class SplitParam(object):
    def __init__(self, K):
        self.K = K
        self.folds = []

def split_trajectories(cfg, param, random_state, data):
    env = cfg.env

    try:
        as_int = env.save_as_int
    except:
        as_int = False
        
    perm = random_state.permutation(data.trajectories)
    np.random.seed(cfg.seed)
    size_of_fold = int((1/param.K)*len(perm))
    for i in range(param.K):
        #np.append(param.folds, np.array([perm[i*size_of_fold:(i+1)*size_of_fold]]))
        param.folds.append(deepcopy(perm[i*size_of_fold:(i+1)*size_of_fold]))
    one_fold_data = []
    multi_fold_data = []
    
    #for i in range(param.K):
    #    one_fold_data.append(deepcopy(data))
    #    multi_fold_data.append(deepcopy(data))

    for i in range(param.K):
        one_fold_data.append(Data(deepcopy(param.folds[i]), env.n_actions, env.n_dim, as_int))
        if param.K > 1:
            if i < param.K - 1:
                multi_fold_data.append(Data(np.concatenate((perm[:i*size_of_fold], perm[(i+1)*size_of_fold:]), axis=0), env.n_actions, env.n_dim, as_int))
            else:
                multi_fold_data.append(Data(perm[:i*size_of_fold], env.n_actions, env.n_dim, as_int))
        else:
            multi_fold_data.append(Data(deepcopy(param.folds[i]), env.n_actions, env.n_dim, as_int))
    return param, one_fold_data, multi_fold_data

def cross_fitting(self, param, random_state, data, cfg, true, aggregate_method='mean'):
    env = cfg.env
    pi_e = cfg.pi_e
    gamma = cfg.gamma

    param, one_fold_data, multi_fold_data = split_trajectories(cfg, param, random_state, data)
    np.random.seed(cfg.seed)
    
    dic = []
    dic_temp = {}

    for i in range(param.K):
        dic_temp.update({'ON POLICY': [float(true), 0]})
        dic.append(deepcopy(dic_temp))

    for model in cfg.models:
        for i in range(param.K):
            # choose one model to supply the Q function for IPS methods
            if 'FQE' == model:
                FQE = FittedQEvaluation()
                FQE.fit(one_fold_data[i], pi_e, cfg, cfg.models[model]['model'])
                FQE_Qs = FQE.get_Qs_for_data(one_fold_data[i], cfg)
                out = self.estimate(FQE_Qs, multi_fold_data[i], gamma, model, true)
                dic[i].update(out)
            elif 'Retrace' == model:
                retrace = Retrace(model, cfg.models[model]['lamb'])
                retrace.fit(one_fold_data[i], pi_e, cfg, cfg.models[model]['model'])
                retrace_Qs = retrace.get_Qs_for_data(one_fold_data[i], cfg)
                out = self.estimate(retrace_Qs, multi_fold_data[i], gamma, model, true)
                dic[i].update(out)
            elif 'Tree-Backup' == model:
                tree = Retrace(model, cfg.models[model]['lamb'])
                tree.fit(one_fold_data[i], pi_e, cfg, cfg.models[model]['model'])
                tree_Qs = tree.get_Qs_for_data(one_fold_data[i], cfg)
                out = self.estimate(tree_Qs, multi_fold_data[i], gamma, model, true)
                dic[i].update(out)
            elif 'Q^pi(lambda)' == model:
                q_lambda = Retrace(model, cfg.models[model]['lamb'])
                q_lambda.fit(one_fold_data[i], pi_e, cfg, cfg.models[model]['model'])
                q_lambda_Qs = q_lambda.get_Qs_for_data(one_fold_data[i], cfg)
                out = self.estimate(q_lambda_Qs, multi_fold_data[i], gamma, model, true)
                dic[i].update(out)
            elif 'Q-Reg' == model:
                q_reg = DirectMethodRegression()
                q_reg.fit(one_fold_data[i], pi_e, cfg, cfg.models[model]['model'])
                q_reg_Qs = q_reg.get_Qs_for_data(one_fold_data[i], cfg)
                out = self.estimate(q_reg_Qs, multi_fold_data[i], gamma, model, true)
                dic[i].update(out)
            elif 'MRDR' == model:
                mrdr = MRDR()
                mrdr.fit(one_fold_data[i], pi_e, cfg, cfg.models[model]['model'])
                mrdr_Qs = mrdr.get_Qs_for_data(one_fold_data[i], cfg)
                out = self.estimate(mrdr_Qs, multi_fold_data[i], gamma, model, true)
                dic[i].update(out)
            elif 'IH' == model:
                # IH is supplied with the whole data
                ih = IH()
                ih.fit(data, pi_e, cfg, cfg.models[model]['model'])
                inf_hor_output = ih.evaluate(data, cfg)
                dic[i].update({'IH': [inf_hor_output, (inf_hor_output - true )**2]})
            elif 'MBased' == model:
                mbased = ApproxModel(cfg, one_fold_data[i].n_actions)
                mbased.fit(one_fold_data[i], pi_e, cfg, cfg.models[model]['model'])
                mbased_Qs = mbased.get_Qs_for_data(pi_e, one_fold_data[i], cfg)
                out = self.estimate(mbased_Qs, multi_fold_data[i], gamma, model, true)
                dic[i].update(out)
            elif 'IS' == model:
                # IS is supplied with the whole data
                out = self.estimate([], data, gamma, model, true, True)
                dic[i].update(out)
            else:
                print(model, ' is not a valid method')
            
    for method in dic[0]:
        aggregate = []
        for i in range(param.K):
            aggregate.append(dic[i][method])
        if 'mean' == aggregate_method:
            dic_temp[method] = np.mean(aggregate, axis=0)
        elif 'median' == aggregate_method:
            dic_temp[method] = np.median(aggregate, axis=0)
    return dic_temp