from abc import ABC, abstractmethod
import random
import sys

from itertools import combinations
from numpy.random import beta, choice
from numpy import median
from math import log
from statistics import mean
import matplotlib.pyplot as plt

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
    
    def __init__(self, *args, **kwargs):
        # records highest and lowest values seen
        self.best  = (-1, -float("inf"))
        self.worst = (-1, float("inf"))

        self.gen_best = (-1, -float("inf"))
        self.gen_worst = (-1, float("inf"))

        self.arms, self.rewards, self.n = self._arms_init(*args, **kwargs)

        self.counts = [0] * len(self.arms)

        self.raw_rewards = []
    
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
        # d is additional data pre-calculated in pre_udpate
        pass
    
    # Selection method for basic multi-play bandit, selecting top n arms to play 
    def play(self, t):
        
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
    
    # Optional
    def pre_update(self, rewards):
        # process list of lists of rewards for each arm
        return rewards

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
        
        self.raw_rewards.append(des)

        des_proc = self.pre_update(des)

        for a, des in enumerate(des_proc):
            if des:
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
        # ((arm, play)), (gen_best, gen_worst), (best,worst), [raw_rewards]
        return tuple((a,p) for a,p in zip(self.arms, self.counts)), (self.gen_best, self.gen_worst), (self.best, self.worst), self.raw_rewards

class MutatorInterfacee(AbstractMutator):
    
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

    # Optional
    # process list of lists of rewards for each arm
    def pre_update(self, rewards):
        return rewards
    
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

        return rates, [(0,0)] * len(rates), plays

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
        p_a, n_a = self.rewards[i]
        n = len([r for r in rewards if r <= 0])
        p = len(rewards) - n
        self.rewards[i] = (p_a+p,n_a+n)

    # Optional but recommended: 
    # Return string representing an arm
    # Given index, arm, count
    def render_arm(self, i, a, c):
        return "pr:{} +-:{} p:{}".format(a, self.rewards[i], c)

class PrMutator(AbstractMutator):
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

        return rates, [(0,0)] * len(rates), [1] if single else plays

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
        pos_r = len([r for r in rewards if r > 0])/len(rewards)
        self.arms[i] += (pos_r - self.arms[i]) * self.lr

        p_a, n_a = self.rewards[i]
        n = len([r for r in rewards if r <= 0])
        p = len(rewards) - n
        self.rewards[i] = (p_a+p,n_a+n)

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
        return "pr: {} +-: {} p: {}".format(a, self.rewards[i], c)

class EpsMutator(AbstractMutator):
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
        # Reward arm i with some function of rewards
        # for easy changing in testing
        f = max
        r = f(rewards)
        print(r)
        self.arms[i] += max(0,r)
        self.rewards[i] += r

class TSMutator(AbstractMutator):
    # Return [arms], [rewards], [n] 
    # where [n] is list of n_plays, chosen randomly uniformly
    # Can also define any args as necessary
    def _arms_init(self, a_0=[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], plays=[1], success=0.2):
        self.success = success # ratio of positives required for arm to be considered successul
        return a_0, [0] * len(a_0), plays 
    
    # Rate a given arm, indexed with i, value of arm a, and time t
    def rate_arm(self, i, a, t):
        p, n = self.arms[i]
        return beta(p+1, n+1)
    
    # Update given index and reward list
    def update(self, i, rewards):
        ratio = len([r for r in rewards if r > 0]) / len(rewards)
        s = 0
        f = 0
        if ratio > self.success:
            s = 1
        else:
            f = 1
        p,n = self.arms[i]

        self.arms[i] = (p+s, n+f)
        self.rewards[i] += median(rewards)

    # Optional but recommended: How to print an arm
    # Given index, arm, count
    def render_arm(self, i, a, c):
        return "a:{} r:{} c:{}".format(a, self.rewards[i], c)

