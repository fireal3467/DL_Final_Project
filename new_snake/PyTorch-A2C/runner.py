from utils import next_state, sample_action, cuda_if
from torch.autograd import Variable
import torch
import gym
import torch.nn.functional as F
from collections import deque

import time

class Runner:
    def __init__(self, datas, hyps, gate_q, stop_q, rew_q):
        """
        hyps - dict object with all necessary hyperparameters
                Keys (Assume string type keys):
                    "gamma" - reward decay coeficient
                    "n_tsteps" - number of steps to be taken in the
                                environment
                    "n_frame_stack" - number of frames to stack for
                                creation of the mdp state
                    "preprocess" - function to preprocess raw observations
                    "env_type" - type of gym environment to be interacted 
                                with. Follows OpenAI's gym api.
        datas - dict of torch tensors with shared memory to collect data. Each 
                tensor contains indices from idx*n_tsteps to (idx+1)*n_tsteps
                Keys (assume string keys):
                    "states" - Collects the MDP states at each timestep t
                    "deltas" - Collects the gae deltas at timestep t+1
                    "rewards" - Collects float rewards collected at each timestep t
                    "dones" - Collects the dones collected at each timestep t
                    "actions" - Collects actions performed at each timestep t
                    If Using Recurrent Model:
                        "h_states" - Collects recurrent states at each timestep t
        gate_q - multiprocessing queue. Allows main process to control when
                rollouts should be collected.
        stop_q - multiprocessing queue. Used to indicate to main process that
                a rollout has been collected.
        rew_q - holds average reward over all processes
                type: multiprocessing Queue
        """

        self.hyps = hyps
        self.datas = datas
        self.gate_q = gate_q
        self.stop_q = stop_q
        self.rew_q = rew_q
        self.obs_deque = deque(maxlen=hyps['n_frame_stack'])


    def run_single(self, net):
        """
        run is the entry function to begin collecting rollouts from the
        environment using the specified net. gate_q indicates when to begin
        collecting a rollout and is controlled from the main process.
        The stop_q is used to indicate to the main process that a new rollout
        has been collected.

        net - torch Module object. This is the model to interact with the
            environment.
        """
        self.net = net
        self.env = gym.make(self.hyps['env_type'])
        state = next_state(self.env, self.obs_deque, obs=None, reset=True,
                                        preprocess=self.hyps['preprocess'])
        self.state_bookmark = state
        self.h_bookmark = None
        if self.net.is_recurrent:
            self.h_bookmark = Variable(cuda_if(torch.zeros(1, self.net.h_size)))
        self.ep_rew = 0
        for p in self.net.parameters(): # Turn off gradient collection
            p.requires_grad = False
        while True:
            self.rollout(self.net, 0, self.hyps)


    def run(self, net):
        """
        run is the entry function to begin collecting rollouts from the
        environment using the specified net. gate_q indicates when to begin
        collecting a rollout and is controlled from the main process.
        The stop_q is used to indicate to the main process that a new rollout
        has been collected.

        net - torch Module object. This is the model to interact with the
            environment.
        """
        self.net = net
        self.env = gym.make(self.hyps['env_type'])
        state = next_state(self.env, self.obs_deque, obs=None, reset=True,
                                        preprocess=self.hyps['preprocess'])
        self.state_bookmark = state
        self.h_bookmark = None
        if self.net.is_recurrent:
            self.h_bookmark = Variable(cuda_if(torch.zeros(1, self.net.h_size)))
        self.ep_rew = 0
        #self.net.train(mode=False) # fixes potential batchnorm and dropout issues
        for p in self.net.parameters(): # Turn off gradient collection
            p.requires_grad = False
        while True:
            idx = self.gate_q.get() # Opened from main process
            self.rollout(self.net, idx, self.hyps)
            self.stop_q.put(idx) # Signals to main process that data has been collected

    def rollout(self, net, idx, hyps):
        """
        rollout handles the actual rollout of the environment for n steps in time.
        It is called from run and performs a single rollout, placing the
        collected data into the shared lists found in the datas dict.

        net - torch Module object. This is the model to interact with the
            environment.
        idx - int identification number distinguishing the
            portion of the shared array designated for this runner
        hyps - dict object with all necessary hyperparameters
                Keys (Assume string type keys):
                    "gamma" - reward decay coeficient
                    "n_tsteps" - number of steps to be taken in the
                                environment
                    "n_frame_stack" - number of frames to stack for
                                creation of the mdp state
                    "preprocess" - function to preprocess raw observations
        """
        state = self.state_bookmark
        h = self.h_bookmark
        n_tsteps = hyps['n_tsteps']
        startx = idx*n_tsteps
        prev_val = None
        for i in range(n_tsteps):
            self.datas['states'][startx+i] = cuda_if(torch.FloatTensor(state))
            state_in = Variable(self.datas['states'][startx+i]).unsqueeze(0)
            if 'h_states' in self.datas:
                self.datas['h_states'][startx+i] = h.data[0]
                h_in = Variable(h.data)
                val, logits, h = net(state_in, h_in)
            else:
                val, logits = net(state_in)
            probs = F.softmax(logits, dim=-1)
            action = sample_action(probs.data)
            action = int(action.item())
            #ME
            time.sleep(0.05)
            #ENDME
            obs, rew, done, info = self.env.step(action+hyps['action_shift'])
            if hyps['render']:
                self.env.render()
            self.ep_rew += rew
            reset = done
            if "Pong" in hyps['env_type'] and rew != 0:
                done = True
            if done:
                self.rew_q.put(.99*self.rew_q.get() + .01*self.ep_rew)
                self.ep_rew = 0
                # Reset Recurrence
                if h is not None:
                    h = Variable(cuda_if(torch.zeros(1,self.net.h_size)))

            self.datas['rewards'][startx+i] = rew
            self.datas['dones'][startx+i] = float(done)
            self.datas['actions'][startx+i] = action
            state = next_state(self.env, self.obs_deque, obs=obs, reset=reset, 
                                                preprocess=hyps['preprocess'])
            if i > 0:
                prev_rew = self.datas['rewards'][startx+i-1]
                prev_done = self.datas['dones'][startx+i-1]
                delta = prev_rew + hyps['gamma']*val.data*(1-prev_done) - prev_val
                self.datas['deltas'][startx+i-1] = delta
            prev_val = val.data.squeeze()

        # Funky bootstrapping
        endx = startx+n_tsteps-1
        if not done:
            state_in = Variable(cuda_if(torch.FloatTensor(state))).unsqueeze(0)
            if 'h_states' in self.datas:
                val, logits, _ = net(state_in, Variable(h.data))
            else:
                val, logits = net(state_in)
            self.datas['rewards'][endx] += hyps['gamma']*val.squeeze() # Bootstrap
            self.datas['dones'][endx] = 1.
        self.datas['deltas'][endx] = self.datas['rewards'][endx] - prev_val
        self.state_bookmark = state
        if h is not None:
            self.h_bookmark = h.data
