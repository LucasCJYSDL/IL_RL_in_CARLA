import numpy as np

def gauss_log_pdf(params, x):
    mean, log_diag_std = params
    d = mean.shape[0]
    cov =  np.square(np.exp(log_diag_std))
    diff = x-mean
    exp_term = -0.5 * np.sum(np.square(diff)/cov, axis=0)
    norm_term = -0.5*d*np.log(2*np.pi)
    var_term = -0.5 * np.sum(np.log(cov), axis=0)
    log_probs = norm_term + var_term + exp_term

    return log_probs


def compute_path_probs(acs, means, logstds):

    horizon = len(acs)
    params = [(means[i], logstds[i]) for i in range(horizon)]
    path_probs = [[gauss_log_pdf(params[i], acs[i])] for i in range(horizon)]

    return np.array(path_probs)


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])

    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]

        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def update_buffer(buffer_dict, seg):
    T = len(seg["rew"])
    for i in range(T):
        direction = int(seg["ob"][i][-1:] - 2)
        buffer_dict[direction].add(seg["ob"][i], seg["ac"][i], seg["lprob"][i], seg["adv"][i], seg["tdlamret"][i], seg["vpred"][i])
