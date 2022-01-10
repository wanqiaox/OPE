 
from tqdm import tqdm
import numpy as np
import os
import json
import pandas as pd
from collections import Counter
from ope.utls.thread_safe import threadsafe_generator


import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

class DataHolder(object):
    def __init__(self, s, a, r, s_, d, policy_action, original_shape):
        self.states = s
        self.next_states = s_
        self.actions = a
        self.rewards = r
        self.dones = d
        self.policy_action = policy_action
        self.original_shape = original_shape

class Data(object):
    def __init__(self, trajectories, n_actions, n_dim, make_int=False):
        self.trajectories = trajectories
        self.n_actions = n_actions
        self.n_dim = n_dim
        self.make_int = make_int

        if self.make_int:
            self.process = lambda x: x.astype('uint8')
        else:
            self.process = lambda x: x

    def __getstate__(self):
        dic = {}
        dic['traj'] = self.trajectories
        dic['n_actions'] = self.n_actions
        dic['n_dim'] = self.n_dim
        dic['make_int'] = self.make_int
        return dic

    def __setstate__(self, dic):
        self.trajectories = dic['traj']
        self.n_actions = dic['n_actions']
        self.n_dim = dic['n_dim']
        self.make_int = dic['make_int']

    def copy(self, low_=None, high_=None):
        if (low_ is not None) and (high_ is not None):
            return Data(self.trajectories[low_:high_], self.n_actions, self.n_dim, self.make_int)
        elif (low_ is not None):
            return Data(self.trajectories[low_:], self.n_actions, self.n_dim, self.make_int)
        elif (high_ is not None):
            return Data(self.trajectories[:high_], self.n_actions, self.n_dim, self.make_int)
        else:
            return Data(self.trajectories, self.n_actions, self.n_dim, self.make_int)

    def bootstrap(self, N):
        idxs = np.random.randint(0, len(self.trajectories), N)
        return Data([self.trajectories[x] for x in idxs], self.n_actions, self.n_dim, self.make_int)

    def frames(self, trajectory_wise=True):
        if trajectory_wise:
            return np.array([data['frames'] for data in self.trajectories])
        else:
            return np.array([data['frames'] for data in self.trajectories]).reshape(-1,1).T

    def states(self, trajectory_wise=True, low_=None, high_=None):

        if low_ is not None and high_ is not None:
            episodes = self.trajectories[low_:high_]
            # pos = np.vstack([np.vstack(x['x']) for x in episodes])
            # N = np.hstack([[low_ + n]*len(x['x']) for n,x in enumerate(episodes)])
            # X = np.array([np.array(self.frames()[int(N[idx])])[pos[idx].astype(int)] for idx in range(len(pos))])
            X = np.array([ self.process(np.array(self.trajectories[low_ + idx]['frames'])[np.array(x)]) for idx, x in enumerate([x['x'] for x in episodes])])
        elif low_ is not None:
            episodes = self.trajectories[low_:]
            # pos = np.vstack([np.vstack(x['x']) for x in episodes])
            # N = np.hstack([[low_ + n]*len(x['x']) for n,x in enumerate(episodes)])
            # X = np.array([np.array(self.frames()[int(N[idx])])[pos[idx].astype(int)] for idx in range(len(pos))])
            X = np.array([ self.process(np.array(self.trajectories[low_ + idx]['frames'])[np.array(x)]) for idx, x in enumerate([x['x'] for x in episodes])])
        elif high_ is not None:
            episodes = self.trajectories[:high_]
            # pos = np.vstack([np.vstack(x['x']) for x in episodes])
            # N = np.hstack([[n]*len(x['x']) for n,x in enumerate(episodes)])
            # X = np.array([np.array(self.frames()[int(N[idx])])[pos[idx].astype(int)] for idx in range(len(pos))])
            X = np.array([ self.process(np.array(self.trajectories[idx]['frames'])[np.array(x)]) for idx, x in enumerate([x['x'] for x in episodes])])
        else:
            X = np.array([ self.process(np.array(self.trajectories[idx]['frames'])[np.array(x)]) for idx, x in enumerate([x['x'] for x in self.trajectories])])

        X = self.process(X)

        if trajectory_wise:
            return X
        else:
            return self.process(np.vstack(X))

    # def states(self, trajectory_wise=True):
    #     if trajectory_wise:
    #         return np.array([data['x'] for data in self.trajectories])
    #     else:
    #         return np.array([data['x'] for data in self.trajectories]).reshape(-1,1).T
    def initial_states(self):
        return self.states()[:,0]

    def actions(self, trajectory_wise=True):
        if trajectory_wise:
            return np.array([data['a'] for data in self.trajectories])
        else:
            return np.array([data['a'] for data in self.trajectories]).reshape(-1,1).T

    def rewards(self, trajectory_wise=True):
        if trajectory_wise:
            return np.array([data['r'] for data in self.trajectories])
        else:
            return np.array([data['r'] for data in self.trajectories]).reshape(-1,1).T

    def next_states(self, trajectory_wise=True, low_=None, high_=None):

        if low_ is not None and high_ is not None:
            episodes = self.trajectories[low_:high_]
            X = np.array([ self.process(np.array(self.trajectories[low_ + idx]['frames'])[np.array(x)]) for idx, x in enumerate([x['x_prime'] for x in episodes])])
        elif low_ is not None:
            episodes = self.trajectories[low_:]
            X = np.array([ self.process(np.array(self.trajectories[low_ + idx]['frames'])[np.array(x)]) for idx, x in enumerate([x['x_prime'] for x in episodes])])
        elif high_ is not None:
            episodes = self.trajectories[:high_]
            X = np.array([ self.process(np.array(self.trajectories[idx]['frames'])[np.array(x)]) for idx, x in enumerate([x['x_prime'] for x in episodes])])
        else:
            X = np.array([ self.process(np.array(self.trajectories[idx]['frames'])[np.array(x)]) for idx, x in enumerate([x['x_prime'] for x in self.trajectories])])

        X = self.process(X)

        if trajectory_wise:
            return X
        else:
            return self.process(np.vstack(X))

    def dones(self, trajectory_wise=True):
        if trajectory_wise:
            return np.array([data['done'] for data in self.trajectories])
        else:
            return np.array([data['done'] for data in self.trajectories]).reshape(-1,1).T

    def base_propensity(self, trajectory_wise=True):
        if trajectory_wise:
            return np.array([data['base_propensity'] for data in self.trajectories])
        else:
            return np.array([data['base_propensity'] for data in self.trajectories]).reshape(-1,1).T

    def target_propensity(self, trajectory_wise=True):
        if trajectory_wise:
            return np.array([data['target_propensity'] for data in self.trajectories])
        else:
            return np.array([data['target_propensity'] for data in self.trajectories]).reshape(-1,1).T

    def next_target_propensity(self, trajectory_wise=True):
        if trajectory_wise:
            return np.array([data['target_propensity'][1:] + [data['extra_propensity']] for data in self.trajectories])
        else:
            return np.array([data['target_propensity'][1:] + [data['extra_propensity']] for data in self.trajectories]).reshape(-1,1).T

    def input_shape(self, process):
        return list(process(np.array(self.trajectories[0]['x'][0])[np.newaxis,...]).shape[1:])

    def num_states(self):
        return len(np.unique([x['frames'] for x in self.trajectories]))

    def ts(self, trajectory_wise=True):
        if trajectory_wise:
            return np.array([range(len(x['x'])) for x in self.trajectories])
        else:
            return np.array([range(len(x['x'])) for x in self.trajectories]).reshape(-1,1).T

    def lengths(self):
        return np.array([len(x['x']) for x in self.trajectories])

    def num_tuples(self):
        return sum(self.lengths())

    def idxs_of_non_abs_state(self):
        dones = self.dones()
        dones = np.hstack([np.zeros((dones.shape[0],1)), dones,])[:,:-1]
        return np.where((1-dones).reshape(-1))[0]

    def value_of_trajectory(self, i, gamma, normalized=False):
        gammas = gamma**np.arange(len(self.trajectories[i]['x']))
        if normalized:
            return np.sum( gammas *  self.trajectories[i]['r'] ),  np.sum( gammas )
        else:
            return np.sum( gammas *  self.trajectories[i]['r'] ), 0

    def value_of_data(self, gamma, normalized=False):
        s, norm = 0, 0
        for i in np.arange(len(self)):
            val, normalization = self.value_of_trajectory(i, gamma, normalized)
            s += val
            norm += normalization

        if normalized:
            return s/norm
        else:
            return s/len(self)

    def __len__(self):
        return len(self.trajectories)

    def all_transitions(self):
        ''' for mle '''
        policy_action = np.vstack([episode['target_propensity'] for episode in self.trajectories])
        dataset = np.hstack([ np.vstack([x['x'] for x in self.trajectories]),
                              np.hstack([x['a'] for x in self.trajectories]).T.reshape(-1, 1),
                              np.hstack([x['r'] for x in self.trajectories]).T.reshape(-1, 1),
                              np.vstack([x['x_prime'] for x in self.trajectories]),
                              np.hstack([x['done'] for x in self.trajectories]).T.reshape(-1, 1),
                              policy_action,
                              np.hstack([[n]*len(x['x']) for n,x in enumerate(self.trajectories)]).T.reshape(-1,1),
                              np.hstack([np.arange(len(x['x'])) for n,x in enumerate(self.trajectories)]).T.reshape(-1,1),])
        return dataset

    def basic_transitions(self):
        ''' for fqe'''
        frames = np.array([x['frames'] for x in self.trajectories])
        data = np.vstack([frames[:,:-1].reshape(-1),
                          np.array([x['a'] for x in self.trajectories]).reshape(-1,1).T,
                          np.array([range(len(x['x'])) for x in self.trajectories]).reshape(-1,1).T,
                          frames[:,1:].reshape(-1),
                          np.array([x['r'] for x in self.trajectories]).reshape(-1,1).T,
                          np.array([x['done'] for x in self.trajectories]).reshape(-1,1).T]).T
        return data

    def omega(self):
        return np.array([[episode['target_propensity'][idx][int(act)]/episode['base_propensity'][idx][int(act)] for idx,act in enumerate(episode['a'])] for episode in self.trajectories])

    def estimate_propensity(self, cfg):
        # WARN: Only works in tabular env with discrete action space. Current implementation is a max likelihood

        if cfg.to_regress_pi_b['model'] == 'tabular':
            data = self.basic_transitions()
            propensity = np.ones((self.n_dim, self.n_actions))/self.n_actions

            df = pd.DataFrame(data[:, [0, 1]], columns=['x','a'])
            terminal = np.max(df['x']) # handle terminal condition

            for (x), group in df.groupby(['x']):
                new_propensity = np.zeros(self.n_actions)
                count_per_action = Counter(group['a'])
                for action, count in count_per_action.items():
                    new_propensity[int(action)] = count/len(group)

                new_propensity += 1e-8
                propensity[int(x)] = new_propensity / sum(new_propensity)

            for episode_num, states in enumerate(np.squeeze(self.states())):
                base_propensity = []
                for state in states:
                    base_propensity.append(propensity[state].tolist())

                self.trajectories[episode_num]['base_propensity'] = base_propensity
        else:

            model = cfg.to_regress_pi_b['model'](self.states()[0][0].shape, self.n_actions)
            optimizer = optim.Adam(model.parameters())

            self.processed_data = self.fill()

            batch_size = cfg.to_regress_pi_b['batch_size']
            dataset_length = self.num_tuples()
            perm = np.random.permutation(range(dataset_length))
            eighty_percent_of_set = int(.8*len(perm))
            training_idxs = perm[:eighty_percent_of_set]
            validation_idxs = perm[eighty_percent_of_set:]
            training_steps_per_epoch = int(np.ceil(len(training_idxs)/float(batch_size)))
            validation_steps_per_epoch = int(np.ceil(len(validation_idxs)/float(batch_size)))

            for k in tqdm(range(cfg.to_regress_pi_b['max_epochs'])):
                
                train_gen = self.generator(training_idxs, fixed_permutation=True, batch_size=batch_size)
                val_gen = self.generator(validation_idxs, fixed_permutation=True, batch_size=batch_size)

                # TODO: earlyStopping, LR reduction

                for step in range(training_steps_per_epoch):
                    
                    with torch.no_grad():
                        (inp, out) = next(train_gen)
                        states = torch.from_numpy(inp).float()
                        actions = torch.from_numpy(out).float().argmax(1)

                    prediction = model.predict_w_softmax(states)
                    
                    loss = nn.NLLLoss()(torch.log(prediction), actions)
                                        
                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.to_regress_pi_b['clipnorm'])
                    optimizer.step()

            for episode_num, states in enumerate(np.squeeze(self.states())):
                base_propensity = []
                for state in states:
                    base_propensity.append(model.predict_w_softmax(torch.from_numpy(state[None,None,...]).float()).detach().numpy()[0].tolist())

                self.trajectories[episode_num]['base_propensity'] = base_propensity

    def fill(self):
        states = self.states()
        states = states.reshape(-1,np.prod(states.shape[2:]))
        actions = self.actions().reshape(-1)
        actions = np.eye(self.n_actions)[actions]

        next_states = self.next_states()
        original_shape = next_states.shape
        next_states = next_states.reshape(-1,np.prod(next_states.shape[2:]))

        policy_action = self.target_propensity().reshape(-1, self.n_actions)
        rewards = self.rewards().reshape(-1)

        dones = self.dones()
        dones = dones.reshape(-1)

        return DataHolder(states, actions, rewards, next_states, dones, policy_action, original_shape)

    @threadsafe_generator
    def generator(self, all_idxs, fixed_permutation=False,  batch_size = 64):
        states = self.processed_data.states
        actions = self.processed_data.actions
        next_states = self.processed_data.next_states
        original_shape = self.processed_data.original_shape
        policy_action = self.processed_data.policy_action
        rewards = self.processed_data.rewards
        dones = self.processed_data.dones
        data_length = len(all_idxs)
        steps = int(np.ceil(data_length/float(batch_size)))

        alpha = 1.
        while True:
            perm = np.random.permutation(all_idxs)
            for batch in np.arange(steps):
                batch_idxs = perm[(batch*batch_size):((batch+1)*batch_size)]

                x = states[batch_idxs].reshape(tuple([-1]) + original_shape[2:])

                acts = actions[batch_idxs]

                yield (x, acts)

def rollout(env, pi_b, process, absorbing_state, pi_e = None, N=10000, T=200, frameskip=1, frameheight=1, path=None, filename='tmp',preprocessor=None, visualize=False, no_op_steps=0, use_only_last_reward=False):
    # filename = os.path.join(path, filename % (N, frameskip))
    # try:
    #     with open(filename) as jsonfile:
    #         trajectories = json.load(jsonfile)
    # except:
    trajectories = []
    for i in tqdm(range(N)):
        done = False
        state = env.reset()

        if no_op_steps > 0:
            for _ in range(3):
                state, _, _, _ = env.step(1) #random action?

        true_state = state[:]
        episode = {'true_state': [],
                   'true_next_state': [],
                   'x': [],
                   'a': [],
                   'r': [],
                   'x_prime': [],
                   'done': [],
                   'base_propensity': [],
                   'target_propensity': [],
                   'frames': [],
                   'extra_propensity': []}
        t = 0
        if preprocessor:
            frames = [preprocessor(np.array([true_state]))]*frameheight #+ [absorbing_state]*(frameheight-1)
            all_frames = [t]*frameheight
            episode['frames'].append(frames[0])
        else:
            frames = [true_state]*frameheight
            all_frames = [t]*frameheight
            episode['frames'].append(state.tolist())
        N_acts = None

        if visualize and (i == 0):
            import matplotlib.pyplot as plt
            #plt.imsave('./videos/enduro/%s_%05d.jpg' % (visualize, t), env.render(mode='rgb_array'))
            plt.imsave('./videos/enduro/%s_%05d.jpg' % (visualize, t), env.render())

        while (t < T): # and (not done):
            # im = env.pos_to_image(np.array(frames)[np.newaxis, ...])
            if not done:

                im = process(np.array(frames)[np.newaxis, ...])
                # im = np.array(frames)[np.newaxis, ...]

                action = int(pi_b.sample(im))#pi_b([state])

                if N_acts is None: N_acts = len(pi_b.predict(im).tolist()[0])
                episode['base_propensity'].append(pi_b.predict(im).tolist()[0])
                if pi_e is not None:
                    episode['target_propensity'].append(pi_e.predict(im).tolist()[0])

                reward = 0
                for _ in range(frameskip):
                    if done:
                        new_state, rew, done = absorbing_state, 0, True
                        continue
                    try:
                        # check if reaching the absorbing state
                        if pi_b.action_map is not None:
                            new_state, rew, done, info = env.step(pi_b.get_action(action))
                        else:
                            new_state, rew, done, info = env.step(action)
                    except:
                        new_state, rew, done, info = env.step(action)
                    reward += rew/frameskip
                    if visualize and (i == 0):
                        plt.imsave('./videos/enduro/%s_%05d.jpg' % (visualize, t), env.render(mode='rgb_array'))
                if use_only_last_reward:
                    reward = rew
                true_state = new_state

            else:
                action = 0
                # propensity = [1/N_acts]*N_acts
                # propensity[-1] = 1 - sum(propensity[:-1])
                # import pdb; pdb.set_trace()
                propensity = [1e-8] * N_acts
                propensity[action] += 1 - sum(propensity)
                episode['base_propensity'].append(propensity)
                if pi_e is not None:
                    episode['target_propensity'].append(propensity)
                new_state, reward, done = absorbing_state, 0, True
                true_state = new_state

            t += 1
            if preprocessor:
                frames.append(preprocessor(np.array([true_state])))
            else:
                frames.append(true_state)
            all_frames += [t]
            x = all_frames[:-1]
            x_ = all_frames[1:]
            all_frames.pop(0)
            frames.pop(0)

            episode['x'].append(x)
            episode['a'].append(action)
            episode['r'].append(reward)
            episode['x_prime'].append(x_)
            episode['done'].append(done)

            # if len(episode['frames'])== 50: import pdb; pdb.set_trace()

            if preprocessor:
                episode['frames'].append(preprocessor(np.array([new_state])))
            else:
                episode['frames'].append(new_state.tolist())

            state = new_state

        episode['frames'][-1] = preprocessor(np.array([absorbing_state])).tolist() if preprocessor else absorbing_state.tolist()
        if pi_e is not None:
            if len(state) > 1:
                if np.all(state == absorbing_state):
                    propensity = [1e-8] * N_acts
                    propensity[action] += 1 - sum(propensity)
                    episode['extra_propensity'] = propensity
                else:
                    im = process(np.array(frames)[np.newaxis, ...])
                    episode['extra_propensity'] = pi_e.predict(im).tolist()[0]
            else:
                if state == absorbing_state:
                    propensity = [1e-8] * N_acts
                    propensity[action] += 1 - sum(propensity)
                    episode['extra_propensity'] = propensity
                else:
                    im = process(np.array(frames)[np.newaxis, ...])
                    episode['extra_propensity'] = pi_e.predict(im).tolist()[0]
        trajectories.append(episode)

        # with open(filename, 'w') as fout:
        #     json.dump(trajectories, fout, indent= 4)
        try:
            as_int = env.save_as_int
        except:
            as_int = False
    return Data(trajectories, env.n_actions, env.n_dim, as_int)
