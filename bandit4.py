from abc import ABC, abstractmethod
import random
import sys

from itertools import combinations
from numpy.random import beta, choice
from numpy import median
from math import log
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

class AbstractMutator(ABC):

    """
    Abstract class for bandit: 
    Requirements: 
        _arms_init(self) # can include extra args if needed
        rate_plays(self, t): return [(play, rating)]
        update(self, arm, rewards)

    Optional:
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
        # Scale s.t. sum of ratings is 1 but 
        # rank arm indices according to ratings, ignoring arms <0
        shuffle_arm = list(range(len(self.arms)))
        random.shuffle(shuffle_arm)
        viable_arms = [i for i in shuffle_arm if arm_ratings[i] >= 0]
        rankings = sorted(viable_arms, key=lambda i: arm_ratings[i])
        # Select top n arms, shuffle to reduce biases due to playing order
        n = random.choice(self.n)
        plays = rankings[:n]
        random.shuffle(plays)

        # Update number of plays 
        for a in plays:
            self.counts[a] += 1

        return plays 
    
    # Called once per generation
    def update_all(self, mu_de):
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
    def _arms_init(self, rates, plays=None, single=False):
        if plays is None:
            plays = [len(rates)]

        if single:
            plays = [1]

        self.single = single
        self.single_arm = -1

        if single:
            t = sum(rates)
            rates = [a/t for a in rates]
        rewards = [(0,0)] * len(rates)
        return rates, rewards, plays

    # Optional: perform any calculations prior to rating arms
    def pre_play(self, t):
        if self.single:
            # pre-select the single arm to play
            r = random.random()
            s = 0 
            for i,a in enumerate(self.arms):
                s += a 
                if r < s: 
                    self.single_arm = i
                    break

    # Rate a given arm, indexed with i, value of arm a, and time t
    def rate_arm(self, i, a, t):
        if self.single:
            return 1 if i == self.single_arm else -1
        return 1 if random.random() < a else -1

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
        return "HProb"

    # Return [arms], [rewards], [n] 
    # where [n] is list of n_plays, chosen randomly uniformly
    # Can also define any args as necessary
    def _arms_init(self, rates, plays=None, lr=0.1, single=False):
        # Recommend plays to be left as is, otherwise arms are unfairly selected
        if plays is None:
            plays = [len(rates)]

        if single:
            plays = [1]

        self.lr = lr

        self.single = single
        self.single_arm = -1

        if single:
            t = sum(rates)
        else:
            t = max(rates)

        rates = [a/t for a in rates]
        return rates, [0] * len(rates), [1] if single else plays

    # Optional: perform any calculations prior to rating arms
    def pre_play(self, t):
        # Adjust arms to max
        if self.single:
            # pre-select the single arm to play
            r = random.random()
            s = 0 
            for i,a in enumerate(self.arms):
                s += a 
                if r < s: 
                    self.single_arm = i
                    return
            
    # Rate a given arm, indexed with i, value of arm a, and time t
    def rate_arm(self, i, a, t):
        if self.single:
            return 1 if i == self.single_arm else -1
        return 1 if random.random() < a else -1

    # Update given index and reward list
    def update(self, i, rewards):
        pos_c = len([r for r in rewards if r > 0])
        range_c = len(rewards)
        ratio = pos_c/range_c

        self.arms[i] += (ratio - self.arms[i]) * self.lr
        
        self.rewards[i] += ratio

    def post_update(self):
        if self.single:
            # Adjust arms to total sum
            t = sum(self.arms)
            self.arms = [a/t for a in self.arms]

        else:
            t = max(self.arms)
            self.arms = [a/t for a in self.arms]

    # Optional but recommended: How to print an arm
    # Given index, arm, count
    def render_arm(self, i, a, c):
        return "pr: {} +/all: {} p: {}".format(a, self.rewards[i], c)

class ProbMutator(HProbMutator):
    # Rewards: (sub(+ve), sum(abs(rewards)))
    def simple_name(self):
        return "Prob"
    
    def update(self, i, rewards):
        try:
            pos_r = sum(r for r in rewards if r > 0)
            range_r = sum(abs(r) for r in rewards)

            ratio = pos_r/range_r

            self.arms[i] += (ratio - self.arms[i]) * self.lr

            self.rewards[i] += ratio
        except Exception as e:
            print(rewards)
            raise e

class EpsMutator(AbstractMutator):
    # Rewards: sum(+ve)/sum(abs(rewards))
    def simple_name(self):
        return "Eps"
    # Return [arms], [rewards], [n] 
    # where [n] is list of desired n, chosen randomly uniformly
    # Can also define any args as necessary
    def _arms_init(self, a_0=[0,0,0,0,0,0], eps=0.2, plays=[1]):
        self.eps = eps
        self.exp = False
        return a_0, [0] * len(a_0), plays

    # Optional: perform any calculations prior to rating arms such as pre-choosing
    def pre_play(self, t):
        self.exp = random.random() < self.eps
    
    # Rate a given arm, indexed with i, value of arm a, and time t
    def rate_arm(self, i, a, t):
        if self.exp:
            return 1
        return a
    
    # Update given index and reward list
    def update(self, i, rewards):
        pos_r = sum(r for r in rewards if r > 0)
        range_r = sum(abs(r) for r in rewards)
        ratio = pos_r/range_r

        self.arms[i] += ratio
        
        self.rewards[i] += ratio
        # # Reward arm i with some function of rewards
        # # for easy changing in testing
        # f = max
        # r = f(rewards)
        # self.arms[i] += max(0,r)
        # self.rewards[i] += r

class HEpsMutator(EpsMutator):
    # Rewards: len(+ve)/len(rewards)
    def simple_name(self):
        return "HEps"
    # Update given index and reward list
    def update(self, i, rewards):
        pos_r = len([r for r in rewards if r > 0])
        range_r = len(rewards)
        ratio = pos_r/range_r

        self.arms[i] += ratio
        self.rewards[i] += ratio

class HTSMutator(AbstractMutator):
    # Rewards: len(+ve)/len(rewards)
    def simple_name(self):
        return "HTS"

    # Return [arms], [rewards], [n] 
    # where [n] is list of n_plays, chosen randomly uniformly
    # Can also define any args as necessary
    def _arms_init(self, a_0=[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], plays=[1]):
        return a_0, [0] * len(a_0), plays 
    
    # Rate a given arm, indexed with i, value of arm a, and time t
    def rate_arm(self, i, a, t):
        s,f = self.arms[i]
        return beta(s+1, f+1)
    
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
        return "TS"
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

