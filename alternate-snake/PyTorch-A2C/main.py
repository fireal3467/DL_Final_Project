from hyperparams import HyperParams
from a2c import A2C
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    a2c_trainer = A2C()
    hyps = dict()
    hyps['exp_name'] = "snake" #"pongbptt"
    hyps['env_type'] = "snake-v0"# "Pong-v0"
    hyps['model_type'] = 'conv'
    hyps['use_bptt'] = True
    hyps['entr_coef'] = .01
    hyps['entr_coef_low'] = .001
    hyps['decay_entr'] = True
    hyps['val_coef'] = .5
    hyps['lr'] = 5e-4
    hyps['lr_low'] = 1e-6
    hyps['decay_lr'] = True
    hyps['gamma'] = .98
    hyps['lambda_'] = .95
    hyps['n_tsteps'] = 32
    hyps['n_rollouts'] = 36
    hyps['n_envs'] = 13
    hyps['max_tsteps'] = 40000000
    hyps['n_frame_stack'] = 3
    hyps['optim_type'] = 'rmsprop'
    hyper_params = HyperParams(hyps)
    a2c_trainer.train(hyper_params.hyps)

