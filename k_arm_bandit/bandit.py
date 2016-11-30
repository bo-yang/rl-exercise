import numpy as np
import matplotlib.pyplot as plt

'''softmax distribution'''
def softmax(prob):
    expEst = np.exp(prob)
    return expEst / np.sum(expEst)

class Bandit:
    def __init__(self, k=10, steps=1000, mu=0):
        self.k = k
        self.steps = steps
        self.est = np.zeros(self.k, dtype='float')     # estimation(Q) of each action
        self.occurs = np.zeros(self.k, dtype='float')  # number of occurrance for each action
        self.rewards = np.zeros(steps, dtype='float')  # average rewards

        self.sigma = 1
        self.mu = mu
        self.values = np.sqrt(self.sigma) * np.random.randn(self.k) + self.mu

        self.best_action = np.argmax(self.values)

    # sample-average method
    def play_sample_average(self, eps=0.1):
        for t in range(0, self.steps):
            # get next action
            a = None
            if np.random.binomial(1, eps) == 1:
                a = np.random.randint(0,self.k)
            else:
                a = np.argmax(self.est)
            # update params
            self.occurs[a] += 1
            reward = np.sqrt(self.sigma) * np.random.randn() + self.values[a]
            self.est[a] += (reward - self.est[a]) / self.occurs[a]
            self.rewards[t] += reward
        return self.rewards

    # Upper_Confidence_Bound Action Selection
    def play_ucb(self, c):
        for t in range(0, self.steps):
            # get next action
            rewards = self.est + c * np.sqrt(np.log(t+1) / (self.occurs + 1))
            a = np.argmax(rewards)
            # update params
            self.occurs[a] += 1
            reward = np.sqrt(self.sigma) * np.random.randn() + self.values[a]
            # TODO: why use a constant step size?
            self.est[a] += 0.1 * (reward - self.est[a])
            self.rewards[t] += reward
        return self.rewards

    # Gradient Bandit Algorithms
    def play_gradient(self, step_size=0.1, use_baseline=True):
        pref = np.zeros(self.k, dtype='float')
        best_actions = np.zeros(self.steps, dtype='float')
        indices = np.arange(0,self.k)
        baseline = 0
        for t in range(0, self.steps):
            # get next action - use random but not argmax
            softmax_pref = softmax(pref)
            a = np.random.choice(indices, p=softmax_pref)
            # update params
            if a == self.best_action:
                best_actions[t] += 1
            reward = np.sqrt(self.sigma) * np.random.randn() + self.values[a]
            if use_baseline: # if baseline is ommitted, then use 0
                baseline = (t * baseline + reward) / float(t+1)
            oneHot = np.zeros(self.k)
            oneHot[a] = 1
            pref += step_size * (reward - baseline) * (oneHot - softmax_pref) 
        return best_actions

num_arm = 10
num_steps = 1000

def sample_average(num_bandits = 2000, epsilons=[0, 0.01, 0.1]):
    for eps in epsilons:
        avg_rewards = np.zeros(num_steps, dtype='float')
        for n in range(0, num_bandits):
            bandit = Bandit(num_arm, num_steps)
            avg_rewards += bandit.play_sample_average(eps)
        avg_rewards /= num_bandits
        plt.plot(avg_rewards, label='epsilon = '+str(eps))
    plt.xlabel('Steps')
    plt.ylabel('average reward')
    plt.legend(loc=4)
    plt.grid()
    plt.show()

def upper_confidence_bound(num_bandits = 1000, c=2, eps=0.1):
    avg_rewards = np.zeros(num_steps, dtype='float')
    ucb_rewards = np.zeros(num_steps, dtype='float')
    for n in range(0, num_bandits):
        bandit_avg = Bandit(num_arm, num_steps)
        avg_rewards += bandit_avg.play_sample_average(eps)
        bandit_ucb = Bandit(num_arm, num_steps)
        ucb_rewards += bandit_ucb.play_ucb(c)
    avg_rewards /= num_bandits
    ucb_rewards /= num_bandits

    plt.plot(avg_rewards, label='$\epsilon$-greedy $\epsilon$ = '+str(eps))
    plt.plot(ucb_rewards, label='UCB c = '+str(c))
    plt.xlabel('Steps')
    plt.ylabel('average reward')
    plt.legend(loc=4)
    plt.grid()
    plt.show()

def gradient_bandit(num_bandits = 200):
    mu = 4
    best_actions = np.zeros([4, num_steps])
    for n in range(0, num_bandits):
        bandit1 = Bandit(num_arm, num_steps, mu) # step size 0.1, with baseline
        best_actions[0] += bandit1.play_gradient(0.1, True)
        bandit2 = Bandit(num_arm, num_steps, mu) # step size 0.1, no baseline
        best_actions[1] += bandit2.play_gradient(0.1, False)
        bandit3 = Bandit(num_arm, num_steps, mu) # step size 0.4, with baseline
        best_actions[2] += bandit3.play_gradient(0.4, True)
        bandit4 = Bandit(num_arm, num_steps, mu) # step size 0.4, no baseline
        best_actions[3] += bandit4.play_gradient(0.4, False)

    labels = ['alpha = 0.1, with baseline',
              'alpha = 0.1, without baseline',
              'alpha = 0.4, with baseline',
              'alpha = 0.4, without baseline']
    for i in range(0, len(best_actions)):
        best_actions[i] /= num_bandits
        plt.plot(best_actions[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend(loc=4)
    plt.grid()
    plt.show()

#sample_average(epsilons=[0, 0.01, 0.1])

upper_confidence_bound(c=2, eps=0.1)

#gradient_bandit()
