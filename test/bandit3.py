from abc import ABC, abstractmethod
import random
import sys

from itertools import combinations
from numpy.random import beta
from math import log

def print_warning(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# FIXME
# FIXME Consider complete overhaul, this isn't working as hoped, perhaps realising what data is helpful to MAB is the key step
# FIXME Consider keeping a record of highest and lowest reward received, allows MAB to scale rewards to 0 and 1 by normalising all the rewards to the min/max value
# FIXME

# TODO self.plays parameter in update is not required, len(rewards) does the same thing

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
        self.best  = (-float("inf"), -1)
        self.worst = (float("inf"),  -1)

        # Initialise arms separately, since all arms have a slightly different representations
        self.arms = []  # these are only included so render_arm and report stop complaining
        self.plays = []

        self._arms_init(*args, **kwargs)
    
    @abstractmethod
    def _arms_init(self):
        pass

    @abstractmethod
    def rate_plays(self, t):
        # return ranking of arms as [(play,rating)] tuples
        # done to support multi-play, rating multiple arms as one "play"
        pass

    def play(self, t):

        play_ratings = self.rate_plays(t)

        _,highest = max(play_ratings, key=lambda x: x[1])
        
        plays = [p for p,r in play_ratings if r >= highest]

        return random.choice(plays)

    def update(self, arm, rewards): 
        # just skimming some general information before asking telling arms to update
        r_max = max(rewards)
        r_min = min(rewards)

        r_b, _ = self.best
        r_w, _ = self.worst

        self.best = (r_max, arm) if r_max > r_b else self.best
        self.worst = (r_min, arm) if r_min < r_w else self.worst
        
        self.arm_update(arm, rewards)

    @abstractmethod
    def arm_update(self, arm, rewards):
        # Update arm with observed rewards
        pass
    
    def render_arm(self, i):
        # return report of arm i as a string
        return "arm:{}, plays{}".format(self.arms[i], self.plays[i])

    def report(self):

        report = ""
        for i in range(len(self.arms)):
            report += "{}: {}\n".format(i, self.render_arm(i))

        report += "\nBest: {}, Worst: {}\n".format(self.best, self.worst)

        return report


class RandomMutator(AbstractMutator):
    def _arms_init(self, rates=[], single_play=True):
        self.arms = rates
        self.scale = sum(rates)
        self.plays = [0] * 6
        self.single = single_play

        self.rewards = [0] * 6

    def rate_plays(self, t):
        if self.single:
            r = random.random()
            tot = 0
            for i,a in enumerate(self.arms):
                tot += a/self.scale
                if tot >= r:
                    return [([i], r)]
        else:
            r = random.random()
            return [([i for i,a in enumerate(self.arms) if a > r], r)]
    
    def arm_update(self, arm, rewards):
        self.rewards[arm] += max(rewards)
        self.plays[arm] += len(rewards)

    def render_arm(self, i):
        return "pr: {}, plays:{}, score:{}".format(self.arms[i], self.plays[i], self.rewards[i])

class EpsMutator(AbstractMutator):
    def _arms_init(self, epsilon=0.1, a_0=None, p_0=None):

        self.arms  = [0] * 6 if a_0 is None else a_0
        self.plays = [0] * 6 if p_0 is None else p_0
        self.eps = epsilon

    def rate_plays(self, t):
        if random.random() < self.eps:
            return [([random.randint(0, len(self.arms)-1)], 1)]
        else:
            return [([a], s) for a,s in enumerate(self.arms)]
    
    def arm_update(self, arm, rewards):
        # since we're trying to be as greedy as possible, just save the highest rewarded arm
        self.arms[arm] += max(rewards)
        self.plays[arm] += len(rewards)
        
    def render_arm(self, i):
        return "score:{}, plays:{}".format(self.arms[i], self.plays[i])

class UCBMutator(AbstractMutator):
    def _arms_init(self):
        self.arms  = [0] * 6
        self.plays = [0] * 6

    def rate_plays(self, t):
        # provide 1 play, increase the plays count
        return []
    
    def arm_update(self, arm, rewards):
        self.arms[arm] += max(rewards)
        

    def render_arm(self, i):
        return ""

class TSMutator(AbstractMutator):
    def _arms_init(self, n_plays=None):
        self.arms = [(0,0)] * 6
        self.plays = [0] * 6
        self.n_plays = [1] if n_plays is None else n_plays

        self.readjust = True
        self.arms_adj = [(0,0)] * 6

    def rate_plays(self, t):
        n = random.choice(self.n_plays)
        # print(self.arms)
        
        if self.readjust:
            self.arms_adj = [(max(s/self.best[0], 0),
                        max(f/-self.worst[0], 0)) 
                        for s,f in self.arms]
            self.readjust = False
            # print(self.arms_adj)
        rate_arms = [beta(s+1,f+1) for s,f in self.arms_adj]

        combos = [list(c) for c in combinations(range(len(self.arms)), n)]

        for c in combos:
            random.shuffle(c)
        
        combo_ratings = [max([rate_arms[a] for a in c]) for c in combos]

        return [(c, r) for c,r in zip(combos, combo_ratings)]

    def arm_update(self, arm, rewards):
        sp, fp = self.arms[arm]

        s = max(rewards)
        f = -min(rewards)

        self.arms[arm] = (sp + s, fp + f)

        self.plays[arm] += len(rewards)

        self.readjust = True

    def render_arm(self, i):
        return "sf:{}, plays:{}".format(self.arms[i], self.plays[i])


"""
class RSMutatorBandit(AbstractBandit): # Random Sampler Bandit, to observe default behaviour

    # Parameters: single_mutation=bool; mutation_probabiities=[float], be sure to include rate for each arm
    def __init__(self, params):
    
        self.single_play = params["single_mutation"]
        
        self.rates = [] # play rates of each arm
        self.arms = [] # observed reward for the arm
        self.plays = [] # number of plays for each arm
        
        for a in params["mutation_rates"]:
            self.rates.append(a)
            self.arms.append(0)
            self.plays.append(0)

    def play(self, round):

        if self.single_play:

            div = max(1,sum(self.rates)) # allows user to specify total mutation rates < 1, allowing for probability of no_mutation

            rand = random.random()
            
            rate = 0

            for i in range(len(self.rates)):

                rate += self.rates[i]

                if rand < rate/div:
                    return [i]
            
            return []
            

        else:
            plays = []
            for i in range(len(self.arms)):
                if random.random() < self.rates[i]:
                    plays.append(i)

            random.shuffle(plays)
            return plays

    # update arms one by one
    def update(self, arm, plays, rewards):
        self.arms[arm] += max(rewards)
        self.plays[arm] += plays

    def report(self):
        for i in range(len(self.rates)):
            print("{}: pr:{}, reward:{}, plays:{}".format(i, self.rates[i], self.arms[i], self.plays[i]))

class EpsMutatorBandit(AbstractBandit): # Epsilon Greedy Bandit

    # Parameters: epsilon=float
    # Optional:   arm_init=[float] 
    def __init__(self, params):
        self.epsilon = params["epsilon"]
        self.arms = []
        self.plays = []

        if "arm_init" in params:
            for arm in params["arm_init"]:
                self.arms.append(arm)
                self.plays.append(0)

        else:
            self.arms = [0] * 6
            self.plays = [0] * 6

    def play(self, round):
        
        if random.random() < self.epsilon:
            # explore
            return [random.randint(0, len(self.arms)-1)]

        # exploit
        best = max(self.arms)
        plays = []

        for i in range(len(self.arms)):
            if self.arms[i] >= best:
                plays.append(i)

        random.shuffle(plays)

        return [plays[0]]

    def update(self, arm, plays, rewards):
        self.arms[arm] += max(rewards)
        self.plays[arm] += plays

    def report(self):
        for i in range(len(self.arms)):
            print("{}: reward:{}, plays:{}".format(i, self.arms[i], self.plays[i]))
    

class MPTSMutatorBandit(AbstractBandit): # Multi-play Thompson Sampling Bandit 

    # Parameters: 
    # Optional:   n_plays=[float], sf_0=[[float,float]]
    # Description: n_plays=list of number of arms to play, picked randomly, defaults to [1]
    #              sf_0=initial success/fail beta distribution for each arm, defaults to (0,0)
    def __init__(self, params):
        
        self.n = []
        self.arms = []
        self.plays = []

        self.highest = -float('inf')
        self.lowest  = float('inf')

        if "n_plays" in params:
            self.n = params["n_plays"]
        else:
            self.n = [1]

        if "sf_0" in params:
            for arm in params["sf_0"]:
                self.arms.append(arm)
                self.plays.append(0)

        else:
            self.arms = [(0,0)] * 6
            self.plays = [0] * 6

    def play(self, round):
        n = random.choice(self.n)
        
        rating = [random.betavariate(s+1,f+1) for (s,f) in self.arms]

        arms = sorted(list(range(0, len(self.arms))), key = lambda a: rating[a])
        # print("played {} arms: {}".format(n, arms[:n]))
        return arms[:n]

    def update(self, arm, plays, rewards):
        # Need to figure out a sensible way to observe reward s, and punishment f; arm=(s,f)
        # s_u = max(0, max(rewards))
        # f_u = max(0, -min(rewards))
        
        s,f = self.arms[arm]

        max_score = max(rewards)
        min_score = min(rewards)

        if max_score < 0 :
            f -= min_score # if max is <0, min has to be <0 too
        elif min_score > 0:
            s += max_score # likewise if min >0, max has to be >0
        else:
            if abs(max_score) > abs(min_score):
                s += max_score
            elif abs(min_score) > abs(max_score):
                f -= min_score

        self.arms[arm] = (s,f)
        self.plays[arm] += plays

    def report(self):
        for i in range(len(self.arms)):
            print("{}: beta:{}, plays:{}".format(i, self.arms[i], self.plays[i]))

class UCBMutatorBandit(AbstractBandit):
    def __init__(self, params):
        
        # TODO implement optional parameters later
        self.arms = [0] * 6
        self.plays = [0] * 6

    def play(self, round):
        
        ratings = [self.arms[i] + (2 * log(round) / self.plays[i])**0.5 if self.plays[i] > 0 else 0 for i in range(len(self.arms))]

        best = max(ratings)
        plays = []

        for i in range(len(self.arms)):
            if self.arms[i] >= best:
                plays.append(i)

        random.shuffle(plays)

        return [plays[0]]

    def update(self, arm, plays, rewards):
        self.arms[arm] += max(rewards)
        self.plays[arm] += plays

    def report(self):
        for i in range(len(self.arms)):
            print("{}: reward:{}, plays:{}".format(i, self.arms[i], self.plays[i]))
"""

""" Beta Implementation
class EpsMutatorBandit(AbstractBandit): # Epsilon greedy 
    def __init__(self, p_0=None):
        self.arms = []
        self.plays = []
        for _ in range(4):
            self.arms.append(0)
            self.plays.append(0)

        try:
            self.epsilon = p_0["epsilon"]
        except:
            print_warning("Problems reading parameters, initialising epsilon as 0.1")
            self.epsilon = 0.1
        
    def play(self, round, context=None):
        
        if random.random() > self.epsilon:
            best_arms = []
            best_score = max(self.arms)

            for arm in range(len(self.arms)):
                if self.arms[arm] >= best_score:
                    best_arms.append(arm)

            return [random.choice(best_arms)]
        
        else: 
            return [random.randint(0, len(self.arms)-1)]

    def update(self, arm, reward, context=None):
        self.arms[arm] += reward
        self.plays[arm] += 1

    def report(self):
        for i in range(len(self.arms)):
            print("{}: {}, {} plays".format(i, self.arms[i], self.plays[i]))


class TSMutatorBandit(AbstractBandit): # Thompson Sampling 
    def __init__(self, p_0=None):
        
        self.arms = []
        self.plays = []

        if p_0 is not None and "arms" in p_0:
            self.arms = p_0["arms"]
            for _ in range(len(p_0["arms"])):
                self.plays.append(0)
        else:
            for _ in range(4):
                self.arms.append([0, 0]) 
                self.plays.append(0)

    def play(self, round, context=None):
        samples = {}
        for a in range(len(self.arms)):
            samples[a] = beta(self.arms[a][0]+1, self.arms[a][1]+1)
        
        return [max(samples, key=lambda x: samples[x])]
    
    def update(self, arm, reward, context=None):
        if reward > 0:
            self.arms[arm][0] += reward
        
        else:
            self.arms[arm][1] -= reward

        self.plays[arm] += 1

    def report(self):
        for i in range(len(self.arms)):
            print("{}: {}, {} plays".format(i, self.arms[i], self.plays[i]))
 
class MPTSMutatorBandit(AbstractBandit):
    def __init__(self, p_0=None):
        
        self.arms = []
        self.plays = []

        self.max_plays = 2

        if p_0 is not None:
            if "arms" in p_0:
                self.arms = p_0["arms"]
                for _ in range(len(p_0["arms"])):
                    self.plays.append(0)
            else:
                default_arms = 4
                for _ in range(default_arms):
                    self.arms.append([0, 0]) 
                    self.plays.append(0)
                
                print("default arms = {}".format(default_arms))

            if "max_plays" in p_0:
                self.max_plays = p_0["max_plays"]
            
            else:
                default_plays = 2
                self.max_plays = default_plays
                print("default plays = {}".format(default_plays))
        else:
            for _ in range(4):
                self.arms.append([0, 0]) 
                self.plays.append(0)
            self.max_plays = 2
            print("default arms, plays = 4, 2")

    def play(self, round, context=None):
        samples = {}
        for a in range(len(self.arms)):
            samples[a] = beta(self.arms[a][0]+1, self.arms[a][1]+1)
        
        return sorted(samples, key=lambda x: samples[x])[:random.randint(1,self.max_plays)]

    def update(self, arm, reward, context=None):
        if reward > 0:
            self.arms[arm][0] += reward
        
        else:
            self.arms[arm][1] -= reward

        self.plays[arm] += 1

    def report(self):
        for i in range(len(self.arms)):
            print("{}: {}, {} plays".format(i, self.arms[i], self.plays[i]))
"""