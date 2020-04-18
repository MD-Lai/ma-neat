from abc import ABC, abstractmethod
import random
import sys

from itertools import combinations
from numpy.random import beta
from math import log
from statistics import mean

def print_warning(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

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

        # TODO consider changing self.plays to self.rewards 
        # Counting plays is identical across all bandits
        self.arms, self.plays, self.n = self._arms_init(*args, **kwargs)

    
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
        # rank arm indices according to ratings, ignoring arms <0
        viable_arms = [i for i in range(len(self.arms)) if arm_ratings[i] > 0]
        rankings = sorted(viable_arms, key=lambda i: arm_ratings[i])
        # Select top n arms, shuffle to reduce biases due to playing order
        n = random.choice(self.n)
        plays = rankings[:n]
        random.shuffle(plays)

        # Update number of plays 
        for a in plays:
            self.plays[a] += 1

        return plays 

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
            
        for a, des in enumerate(des):
            self.update(a, des)

    @abstractmethod
    def update(self, arm, rewards):
        # Update arm with observed rewards
        pass
    
    # Optional but recommended 
    def render_arm(self, i, a, p):
        # return report of arm i as a string
        return "arm: {} plays: {}".format(a, p)

    def report(self):

        report = "\n{}\n".format(type(self).__name__)
        for i,a_p in enumerate(zip(self.arms, self.plays)):
            a,p = a_p
            report += "{}: {}\n".format(i, self.render_arm(i,a,p))

        report += "\nGenerational best: {}\nGenerational worst: {}\n".format(self.gen_best, self.gen_worst)
        report += "\nBest reward: {}\nWorst reward: {}\n".format(self.best, self.worst)

        return report

    def get_data(self):
        # ((arm, play)), (gen_best, gen_worst), (best,worst)
        return tuple((a,p) for a,p in zip(self.arms, self.plays)), (self.gen_best, self.gen_worst), (self.best, self.worst)

class RandomMutator(AbstractMutator):

    # Return ([arms], [plays], [n]), can use as additional set up
    def _arms_init(self, rates):
        self.scale = sum(rates)
        self.play_pr = 0
        self.rewards = [(0,0)] * 6
        return (rates, [0] * len(rates), [len(rates)])

    # Return rating of a given index, arm, and time step
    def rate_arm(self, i, arm, t):
        return 1 if random.random() < arm else -1
    
    def update(self, arm, rewards):
        neg,pos = self.rewards[arm]
        self.rewards[arm] = (neg+len([r for r in rewards if r<0]), pos+len([r for r in rewards if r>0]))
    
    def render_arm(self, i, a, p):
        neg,pos = self.rewards[i]
        rat = float("inf") if neg <= 0 else pos/neg
        return "pr: {} neg,pos:({},{}) rat: {} plays: {}".format(a, neg,pos, rat, p)

class PRMutator(AbstractMutator):
    # Return ([arms], [plays], [n]), can use as additional set up
    def _arms_init(self, rates, lr=0.1, min_rate=0.01):
        self.play_pr = 0
        self.rewards = [(0,0)] * len(rates)
        self.lr = 0.1
        self.min_rate = min_rate
        return (rates, [0] * len(rates), [len(rates)])

    def pre_play(self, t):
        # adjust 
        max_arm = max(self.arms)
        self.adj = [a/max_arm for a in self.arms]

    # Return rating of a given index, arm, and time step
    def rate_arm(self, i, arm, t):
        return 1 if random.random() < self.adj[i] else 0
    
    def update(self, arm, rewards):
        # FIXME Update doesn't work conceptually, rates drop too fast
        # Because it should'e been the difference that we're adding
        if len(rewards) > 0:
            neg,pos = self.rewards[arm]

            neg_r = [r for r in rewards if r<0]
            pos_r = [r for r in rewards if r>0]

            neg_n = max(1,len(neg_r))
            pos_n = max(1,len(pos_r))

            ratio = pos_n/(pos_n+neg_n)
            # print(ratio)

            new_pr = self.arms[arm] + (ratio-self.arms[arm]) * self.lr
            self.arms[arm] = new_pr if new_pr > self.min_rate else self.min_rate

            self.rewards[arm] = (neg+neg_n, pos+pos_n)


    def render_arm(self, i, a, p):
        neg,pos = self.rewards[i]
        rat = float("inf") if neg <= 0 else pos/neg
        return "pr: {} neg,pos:({},{}) rat: {} plays: {}".format(a, neg,pos, rat, p)


class EpsMutator(AbstractMutator):
    # Return ([arms], [plays], [n]), can use as additional set up
    def _arms_init(self, a_0=None, n=None, eps=0.2):
        self.e = eps
        arm = [0] * 6 if a_0 is None else a_0
        p = [0] * 6
        plays = [1] if n is None else [range(len(arm))]
        return arm, p, plays

    # Calculate and store any calculations repeated for each arm
    # Called once per play
    def pre_play(self, t):
        self.explore = random.random() < self.e

    # Return rating of a given index, arm, and time step
    # Called once per arm
    def rate_arm(self, i, arm, t):
        if self.explore:
            return 1
        return arm
    
    # Update a given arm with given reward, no need to count n-plays
    def update(self, arm, rewards):
        self.arms[arm] += sum(rewards)/len(rewards)
    
# UCB has problems that make it not great for NEAT
class UCBMutator(AbstractMutator):
    # Return ([arms], [plays], [n]), can use as additional set up
    def _arms_init(self):
        pass

    # Calculate and store any calculations repeated for each arm
    # Called once per play 
    def pre_play(self, t):
        pass
    # Return rating of a given index, arm, and time step
    # Called once per arm
    def rate_arm(self, i, arm, t):
        pass
    
    # Update a given arm with given reward, no need to count n-plays
    def update(self, arm, rewards):
        pass
    
    # Return string describing an arm
    def render_arm(self, i, a, p):
        pass

class TSMutator(AbstractMutator):
    # Return ([arms], [plays], [n]), can use as additional set up
    def _arms_init(self,a_0=None, n=None):
        arms = [(0,0)] * 6 if a_0 is None else a_0
        plays = [0] * 6
        n_plays = [1] if n is None else n
        return (arms, plays, n_plays)

    # Calculate and store any calculations repeated for each arm
    # Called once per play
    def pre_play(self, t):
        # self.adj = [(max(s/self.best[0], 0),
        #              max(f/-self.worst[0], 0))  for (s,f) in self.arms]
        pass

    # Return rating of a given index, arm, and time step
    # Called once per arm
    def rate_arm(self, i, arm, t):
        s,f = self.arms[i] 
        return beta(s+1,f+1)
    
    # Update a given arm with given reward, no need to count n-plays
    def update(self, arm, rewards):
        if len(rewards) > 0:
            s_a, f_a = self.arms[arm]
            s = 1 if max(rewards) >= self.best[1] else 0
            f = 1 if min(rewards) <= self.worst[1] else 0

            self.arms[arm] = (s_a+s , f_a+f)
    
    # Return string describing an arm
    def render_arm(self, i, a, p):
        return "sf: {}, plays: {}".format(a, p)
