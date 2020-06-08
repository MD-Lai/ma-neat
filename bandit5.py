from abc import ABC, abstractmethod
import random
import sys

from itertools import combinations
from numpy.random import beta, choice
from numpy import median
from math import log, exp
from statistics import mean

import matplotlib.pyplot as plt
import copy
import pickle

import helper

# TODO Base class decides what arm to play
# TODO rate_arm function to describe how to score a given arm 
# TODO play to select how many arms to play
# TODO update just updates score of given arm
# TODO make reward aggregate selectable somehow

"""
this is a stupid and needlessly complicated system, just do play and update next time
pre_play and rate_arms is so useless and complicated to accomodate all bandits.
I don't even think it can work as a normal bandit thanks to the update scheme 
"""

class AbstractMutator(ABC):

    """
    Abstract class for bandit: 
    Requirements: 
        _arms_init(self) # can include extra args if needed
        rate_plays(self, t): return [(play, rating)]
        update(self, arm, rewards)

    Optional:
        pre_play(self,t)
        render_arm(self, i): return string_representing_arm
    """
    @abstractmethod
    def simple_name(self):
        pass
    
    def __init__(self, *args, **kwargs):
        # records highest and lowest values seen
        self.best  = (-1, -float("inf"))
        self.worst = (-1, float("inf"))

        self.gen_best = (-1, -float("inf"))
        self.gen_worst = (-1, float("inf"))

        self.arms, self.rewards, self.n = self._arms_init(*args, **kwargs)

        self.counts = [0] * len(self.arms)

        self.gens = 0
    
    @abstractmethod
    def _arms_init(self, *args, **kwargs):
        # Return initial parameters of arms and plays ([arms],[plays], [n])
        # Can use as additional set up
        pass

    # Optional
    def pre_play(self, t):
        # Calculate anything before initiating ratings
        pass 

    @abstractmethod
    def rate_arm(self, i, arm, t):
        # Return score of a given arm
        # t is time instance
        pass
    
    # Selection method for basic multi-play bandit, selecting top n arms to play 
    def play(self, t):
        
        # Record arms and rewards
        # self.arm_history.append(copy.deepcopy(self.arms))
        # self.reward_history.append(copy.deepcopy(self.rewards))

        self.pre_play(t)
        # Generate rating for all arms, <0 indicates don't consider arm
        arm_ratings = [self.rate_arm(i, a, t) for i,a in enumerate(self.arms)]
        # normalise arm ratings s.t. it sums to 1
        arm_sum = sum(arm_ratings)
        arm_ratings = [1/len(self.arms) if arm_sum == 0 else a/arm_sum for a in arm_ratings]
        # print(arm_ratings)
        # select i's weighted on arm ratings
        # also randomly select n plays
        n = choice(self.n)
        plays = choice(range(len(self.arms)), size=n, replace=False, p=arm_ratings)
        
        # print(plays)

        # Update number of plays 
        for a in plays:
            self.counts[a] += 1

        return plays 
    
    # Called once per generation
    def update_all(self, mu_de):
        self.gens += 1
        self.gen_best = max(mu_de, key=lambda m_d: m_d[1])
        self.gen_worst = min(mu_de, key=lambda m_d: m_d[1])

        if self.gen_best[1] > self.best[1]:
            self.best = self.gen_best
        if self.gen_worst[1] < self.worst[1]:
            self.worst = self.gen_worst

        des = [[] for _ in range(len(self.arms))]

        for mu, de in mu_de:
            des[mu].append(de)
        
        # self.raw_rewards.append(des)

        for a, des in enumerate(des):
            # if des:
            if sum(abs(de) for de in des) > 0 and len(des) > 0:
                self.update(a, des)

        self.post_update()

    @abstractmethod
    def update(self, arm, rewards):
        # Update arm with observed rewards
        pass
    
    # Optional
    def post_update(self):
        # Perform any post processing as required
        pass

    # Optional but recommended 
    def render_arm(self, i, a, c):
        # return report of arm i as a string
        return "arm:{} count:{}".format(a, c)

    def report(self):

        report = "\n{}\n".format(type(self).__name__)
        for i,a_p in enumerate(zip(self.arms, self.counts)):
            a,p = a_p
            report += "{}: {}\n".format(i, self.render_arm(i,a,p))

        report += "\nGenerational best: {}\nGenerational worst: {}\n".format(self.gen_best, self.gen_worst)
        report += "\nBest reward: {}\nWorst reward: {}\n".format(self.best, self.worst)

        return report

    def get_data(self):
        # ((arm, play)), (generational_best, generational_worst), (overall_best, overall_worst) 
        # return tuple(zip(self.arms, self.rewards, self.counts)), (self.gen_best, self.gen_worst), (self.best, self.worst)

        # no reason to complicatedly bunch them up
        # Most important to look at is how arms develops over time and counts ends up, rewards is a bit ehh
        return self.arms, self.rewards, self.counts, self.gen_best, self.gen_worst, self.best, self.worst

class MutatorInterfacee(AbstractMutator):

    # Return simple version of name to use for records externally
    def simple_name(self):
        pass
    
    # Return [arms], [rewards], [n] 
    # where [n] is list of n_plays, chosen randomly uniformly
    # Can also define any args as necessary
    def _arms_init(self):
        pass

    # Optional: perform any calculations prior to rating arms such as pre-choosing
    def pre_play(self, t):
        pass
    
    # Rate a given arm, indexed with i, value of arm a, and time t
    def rate_arm(self, i, a, t):
        pass

    # Update given index and reward list
    def update(self, i, rewards):
        pass

    # Optional
    # Perform any post processing as required
    def post_update(self):
        pass

    # Optional but recommended: How to print an arm
    # Given index, arm, count
    def render_arm(self, i, a, c):
        pass

class RandomMutator(AbstractMutator):
    
    def simple_name(self):
        return "Random"
    # Return [arms], [rewards], [n] 
    # where [n] is list of n_plays, chosen randomly uniformly
    # Can also define any args as necessary
    def _arms_init(self, rates, plays=[1]):
        rewards = [(0,0) for _ in range(len(rates))]
        return rates, rewards, plays

    # Rate a given arm, indexed with i, value of arm a, and time t
    def rate_arm(self, i, a, t): 
        return a

    # Update given index and reward list
    def update(self, i, rewards):
        p_c, p_t = self.rewards[i]

        p = len([r for r in rewards if r > 0])
        c = len(rewards)

        pr = sum(r for r in rewards if r > 0)
        tot = sum(abs(r) for r in rewards)

        # Save both the heuristic reward and non-heuristic
        self.rewards[i] = (p_c+(p/c), p_t+(pr/tot))

    # Optional but recommended: 
    # Return string representing an arm
    # Given index, arm, count
    def render_arm(self, i, a, c):
        return "pr:{} pos_c,count,pos_r,tot_r:{} p:{}".format(a, self.rewards[i], c)

class HProbMutator(AbstractMutator):
    # Rewards: (len(+ve), len(rewards))
    def simple_name(self):
        return "H-Prob"

    # Return [arms], [rewards], [n] 
    # where [n] is list of n_plays, chosen randomly uniformly
    # Can also define any args as necessary
    def _arms_init(self, rates=[0,0,0,0,0,0], plays=[1], lr=0.1):
        # Recommend plays to be left as is, otherwise arms are unfairly selected

        self.lr = lr

        self.single_arm = -1

        a = [exp(r/self.lr) for r in rates]
        a_s = sum(a)
        arms = [a/a_s for a in a]

        # e^(avg(reward)/tau)

        return arms, [0] * len(arms), plays

    # Optional: perform any calculations prior to rating arms
    def pre_play(self, t):
        # e^(avg(reward)/tau)
        if t > 0:
            a = [exp((r/t)/self.lr) for r in self.rewards]
            a_s = sum(a)
            self.arms = [a/a_s for a in a]
            
    # Rate a given arm, indexed with i, value of arm a, and time t
    def rate_arm(self, i, a, t):
        return a

    # Update given index and reward list
    def update(self, i, rewards):
        pos_c = len([r for r in rewards if r > 0])
        range_c = len(rewards)
        ratio = pos_c/range_c

        # TODO
        # self.arms[i] += (ratio - self.arms[i]) * self.lr
        
        self.rewards[i] += ratio
    

    # Optional but recommended: How to print an arm
    # Given index, arm, count
    def render_arm(self, i, a, c):
        return "pr: {} +ve/all: {} p: {}".format(a, self.rewards[i], c)

class ProbMutator(HProbMutator):
    # Rewards: (sub(+ve), sum(abs(rewards)))
    def simple_name(self):
        return "N-Prob"
    
    def update(self, i, rewards):
        pos_r = sum(r for r in rewards if r > 0)
        range_r = sum(abs(r) for r in rewards)

        ratio = pos_r/range_r

        self.rewards[i] += ratio

class EpsMutator(AbstractMutator):
    # Rewards: sum(+ve)/sum(abs(rewards))
    def simple_name(self):
        return "N-Eps"
    # Return [arms], [rewards], [n] 
    # where [n] is list of desired n, chosen randomly uniformly
    # Can also define any args as necessary
    def _arms_init(self, a_0=[0,0,0,0,0,0], eps=0.2, plays=[1]):
        self.eps = eps
        self.exp = False
        self.arm_max = 0
        arm = [a for a in a_0]
        return arm, [0] * len(arm), plays

    # Optional: perform any calculations prior to rating arms such as pre-choosing
    def pre_play(self, t):
        self.exp = random.random() < self.eps
        for i,r in enumerate(self.rewards):
            self.arms[i] = r/(t+1)

    # Rate a given arm, indexed with i, value of arm a, and time t
    def rate_arm(self, i, a, t):
        if self.exp:
            return 1
        return 1 if a == max(self.arms) else 0
    
    # Update given index and reward list
    def update(self, i, rewards):
        pos_r = sum(r for r in rewards if r > 0)
        range_r = sum(abs(r) for r in rewards)
        ratio = pos_r/range_r

        # self.arms[i] += ratio
        
        self.rewards[i] += ratio
        # # Reward arm i with some function of rewards
        # # for easy changing in testing
        # f = max
        # r = f(rewards)
        # self.arms[i] += max(0,r)
        # self.rewards[i] += r

    def post_update(self):
        print(self.arms)

class HEpsMutator(EpsMutator):
    # Rewards: len(+ve)/len(rewards)
    def simple_name(self):
        return "H-Eps"
    # Update given index and reward list
    def update(self, i, rewards):
        pos_r = len([r for r in rewards if r > 0])
        range_r = len(rewards)
        ratio = pos_r/range_r

        # self.arms[i] += ratio
        self.rewards[i] += ratio

class HTSMutator(AbstractMutator):
    # Rewards: len(+ve)/len(rewards)
    def simple_name(self):
        return "H-TS"

    # Return [arms], [rewards], [n] 
    # where [n] is list of n_plays, chosen randomly uniformly
    # Can also define any args as necessary
    def _arms_init(self, a_0=[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], plays=[1]):
        self.eligible = -1
        return a_0, [0] * len(a_0), plays 

    def pre_play(self, t):
        samples = [(a,beta(s+1, f+1)) for a,(s,f) in enumerate(self.arms)]
        a, _ = max(samples, key=lambda ab: ab[1])
        self.eligible = a
        
    
    # Rate a given arm, indexed with i, value of arm a, and time t
    def rate_arm(self, i, a, t):
        return 1 if i == self.eligible else 0
    
    # Update given index and reward list
    def update(self, i, rewards):
        s,f = self.arms[i]

        pos_c = len([r for r in rewards if r > 0]) 
        tot_c = len(rewards)
        ratio = pos_c/tot_c

        s_a = ratio
        f_a = 1-ratio

        self.arms[i] = (s_a+s, f_a+f)
        self.rewards[i] += ratio

    # Optional but recommended: How to print an arm
    # Given index, arm, count
    def render_arm(self, i, a, c):
        return "a:{} r:{} c:{}".format(a, self.rewards[i], c)

class TSMutator(HTSMutator):
    # Rewards: sum(+ve)/sum(abs(rewards))
    def simple_name(self):
        return "N-TS"
    # Update given index and reward list
    def update(self, i, rewards):
        s,f = self.arms[i]

        pos_c = sum([r for r in rewards if r > 0])
        tot_c = sum(abs(r) for r  in rewards)

        ratio = pos_c/tot_c

        s_a = ratio
        f_a = 1-ratio

        self.arms[i] = (s_a+s, f_a+f)
        self.rewards[i] += ratio

