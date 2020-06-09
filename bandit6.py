from numpy.random import beta, choice
import random
import copy
import math

class BanditInterface:
    def simple_name(self):
        return "Interface"

    def __init__(self):
        # can and should be defined to take different args
        # initiate arms, must have these fields
        self.arms = []
        self.rewards = []
        self.counts = []

    def play(self, t):
        # Select index of action at time t and maybe context
        # multi-play bandit will return a list of arms but needs to be written to accomodate that
        # specific n can be included in context, up to caller to expect and use list
        # return choice(range(len(self.arms)))
        pass
    
    def update(self, a, r):
        # update arm a with given reward r
        # increment count only if it receives rewards
        # self.arms[a] += r
        # self.rewards[a] += r
        # self.counts[a] += 1
        pass 

    def update_batch(self, arm_rewards):
        # potential use, update an arm with a collection of rewards
        # self.arms[a] += sum(rs)/len(rs)
        # self.rewards[a] += sum(rs)
        # self.counts[a] += len(rs)
        pass

    def get_data(self):
        return copy.deepcopy(self.arms), copy.deepcopy(self.rewards), copy.deepcopy(self.counts)

class Static(BanditInterface):
    def simple_name(self):
        return "Static"

    def __init__(self, chances):
        self.arms = [c/sum(chances) for c in chances]
        self.rewards = [((0,0),(0,0)) for _ in chances]
        self.counts = [0 for _ in chances]

    def play(self, t):
        a = choice(range(len(self.arms)), p=self.arms)
        return a

    def update(self, a, r):
        (p_N, n_N), (p_H, n_H) = self.rewards[a]
        if r > 0:
            p_N += r
            p_H += 1
        else:
            n_N += abs(r)
            n_H += 1
        self.rewards[a] = (p_N, n_N), (p_H, n_H)
        self.counts[a] += 1

    def update_batch(self, arm_rewards):
        # Update bandits with a list of arm,reward pairs
        for a,r in arm_rewards:
            self.update(a,r)

# Track positive changes and negative changes
# Each arm has a (positive, negative) tuple, 
# Mean reward is calculated as positive/(positive+negative)
# Therefore Numeric sums up delta fitnesses 
# Heuristic counts the number of rewards
# N-Softmax
class N_Softmax(BanditInterface):
    def simple_name(self):
        return "N-Softmax"

    def __init__(self, n, tau): # don't care about initial parameters, just set it all to 0
        self.tau = tau
        self.rewards = [(0,0) for _ in range(n)]
        arms_exp = [math.exp((p/(p+n or 1))/self.tau) for p,n in self.rewards]
        self.arms = [a/sum(arms_exp) for a in arms_exp]
        self.counts = [0 for _ in range(n)]

    def play(self, t):
        return choice(range(len(self.arms)), p=self.arms)

    def update(self, a, r):
        p,n = self.rewards[a]
        if r <= 0:
            n += abs(r)
        else:
            p += r
        self.rewards[a] = (p,n)
        self.counts[a] += 1
        for a in range(len(self.arms)):
            arms_exp = [math.exp((p/(p+n or 1))/self.tau) for p,n in self.rewards]
            self.arms[a] = arms_exp[a]/sum(arms_exp)

    def update_batch(self, arm_rewards):
        for a,r in arm_rewards:
            p,n = self.rewards[a]
            if r <= 0:
                n += abs(r)
            else:
                p += r
            self.rewards[a] = (p,n)
            self.counts[a] += 1

        for a in range(len(self.arms)):
            arms_exp = [math.exp((p/(p+n or 1))/self.tau) for p,n in self.rewards]
            self.arms[a] = arms_exp[a]/sum(arms_exp)

# H-Softmax
class H_Softmax(N_Softmax):
    def simple_name(self):
        return "H-Softmax"
    def update(self, a, r):
        p,n = self.rewards[a]
        if r <= 0:
            n += 1
        else:
            p += 1
        self.rewards[a] = (p,n)
        self.counts[a] += 1
        for a in range(len(self.arms)):
            arms_exp = [math.exp((p/(p+n or 1))/self.tau) for p,n in self.rewards]
            self.arms[a] = arms_exp[a]/sum(arms_exp)

    def update_batch(self, arm_rewards):
        for a,r in arm_rewards:
            p,n = self.rewards[a]
            if r <= 0:
                n += 1
            else:
                p += 1
            self.rewards[a] = (p,n)
            self.counts[a] += 1

        for a in range(len(self.arms)):
            arms_exp = [math.exp((p/(p+n or 1))/self.tau) for p,n in self.rewards]
            self.arms[a] = arms_exp[a]/sum(arms_exp)

# N-Eps
class N_Eps(BanditInterface):
    def simple_name(self):
        return "N-Eps"
    def __init__(self, n, eps):
        self.eps = eps
        self.arms = [0 for _ in range(n)]
        self.rewards = [(0,0) for _ in range(n)]
        self.counts = [0 for _ in range(n)]

    def play(self, t):
        if random.random() < self.eps:
            return choice(range(len(self.arms)))

        r = max(self.arms)
        viable = [i for i in range(len(self.arms)) if self.arms[i] == r]
        return choice(viable)
    
    def update(self, a, r):
        p,n = self.rewards[a]

        if r > 0:
            p += r
        else:
            n += abs(r)

        self.rewards[a] = (p,n)
        self.counts[a] += 1

        self.arms[a] = p/((p+n) or 1)

    def update_batch(self, arm_rewards):
        for a,r in arm_rewards:
            p,n = self.rewards[a]

            if r > 0:
                p += r
            else:
                n += abs(r)

            self.rewards[a] = (p,n)
            self.counts[a] += 1
        
        for a in range(len(self.arms)):
            p,n = self.rewards[a]
            self.arms[a] = p/((p+n) or 1)

# H-Eps
class H_Eps(N_Eps):
    def simple_name(self):
        return "H-Eps"
    def update(self, a, r):
        p,n = self.rewards[a]

        if r > 0:
            p += 1
        else:
            n += 1

        self.rewards[a] = (p,n)
        self.counts[a] += 1

        self.arms[a] = p/((p+n) or 1)

    def update_batch(self, arm_rewards):
        for a,r in arm_rewards:
            p,n = self.rewards[a]

            if r > 0:
                p += 1
            else:
                n += 1

            self.rewards[a] = (p,n)
            self.counts[a] += 1
        
        for a in range(len(self.arms)):
            p,n = self.rewards[a]
            self.arms[a] = p/((p+n) or 1)

# Thompson Sampling is the only one with different reward structure
# N-TS
class N_TS(BanditInterface):
    def simple_name(self):
        return "N-TS"
    def __init__(self, n):
        self.arms = [0 for _ in range(n)]
        self.rewards = [(0,0) for _ in range(n)]
        self.counts = [0 for _ in range(n)]

    def play(self, t):
        # average positive and average negative, boosted by the number of times played 
        rewards_adj = [(c*p/((p+n) or 1), c*n/((p+n) or 1)) for (p,n), c in zip(self.rewards, self.counts)]
        theta = [beta(p+1,n+1) for p,n in rewards_adj]
        a,_ = max(enumerate(theta), key=lambda a_t: a_t[1])
        return a

    def update(self, a, r):
        p,n = self.rewards[a]
        
        if r > 0:
            p += r
        else: 
            n += abs(r)
        
        self.rewards[a] = (p,n)
        self.counts[a] += 1
        self.arms[a] = p/(n+p or 1)

    def update_batch(self, arm_rewards):
        for a,r in arm_rewards:
            p,n = self.rewards[a]
            if r > 0:
                p += r
            else: 
                n += abs(r)
            
            self.rewards[a] = (p,n)
            self.counts[a] += 1
        
        for a in range(len(self.arms)):
            p,n = self.rewards[a]
            self.arms[a] = p/(n+p or 1)

# H-TS
class H_TS(N_TS):
    def simple_name(self):
        return "H-TS"
    def play(self, t):
        # average positive and average negative, boosted by the number of times played 
        theta = [beta(p+1,n+1) for p,n in self.rewards]
        a,_ = max(enumerate(theta), key=lambda a_t: a_t[1])
        return a

    def update(self, a, r):
        p,n = self.rewards[a]
        
        if r > 0:
            p += 1
        else: 
            n += 1
        
        self.rewards[a] = (p,n)
        self.counts[a] += 1
        self.arms[a] = p/(n+p or 1)

    def update_batch(self, arm_rewards):
        for a,r in arm_rewards:
            p,n = self.rewards[a]
            if r > 0:
                p += 1
            else: 
                n += 1
            
            self.rewards[a] = (p,n)
            self.counts[a] += 1
        
        for a in range(len(self.arms)):
            p,n = self.rewards[a]
            self.arms[a] = p/(n+p or 1)

if __name__ == "__main__":
    import sys
    # unit test each bandit
    _, tst = sys.argv
    arms = 6
    reward_prs = [0.2, 0.5, 0.1, 0.6, 0.7, 0.01]
    reward_prs = [0.1, 0.1, 0.8, 0.1, 0.9, 0.1]
    rounds = 2000

    if tst == "Static":
        bdt = Static([0.1,0.2,0.3,0.4,0.5,0.6])
    elif tst == "N_Softmax":
        bdt = N_Softmax(6, 0.2)
    elif tst == "H_Softmax":
        bdt = H_Softmax(6, 0.2)
    elif tst == "N_Eps":
        bdt = N_Eps(6, 0.1)
    elif tst == "H_Eps":
        bdt = H_Eps(6,0.1)
    elif tst == "N_TS":
        bdt = N_TS(6)
    elif tst == "H_TS":
        bdt = H_TS(6)

    # for i in range(rounds):
    #     chosen = bdt.play(i)

    #     reward = 1 if random.random() < reward_prs[chosen] else -1

    #     bdt.update(chosen, reward)

    #     a, r, c = bdt.get_data()
    #     print(f"choice {chosen}")
    #     print(f"reward {reward}")
    #     print(f"arms   {a}")
    #     print(f"rews   {r}")
    #     print(f"counts {c}")
    
    for i in range(rounds):
        batch_rewards = []
        for i in range(50):
            chosen = bdt.play(i)
            reward = 1 if random.random() < reward_prs[chosen] else -1
            batch_rewards.append((chosen,reward))

        bdt.update_batch(batch_rewards)

        a, r, c = bdt.get_data()
        print(f"choice {chosen}")
        print(f"reward {reward}")
        print(f"arms   {a}")
        print(f"rews   {r}")
        print(f"counts {c}")
    print(reward_prs)