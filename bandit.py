from abc import ABC, abstractmethod
import random
import sys

from numpy.random import beta

def print_warning(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# FIXME
# FIXME Consider complete overhaul, this isn't working as hoped, perhaps realising what data is helpful to MAB is the key step
# FIXME

class AbstractBandit(ABC):

    @abstractmethod
    def __init__(self, params):
        # Initialise arms with optional parameters, passed as a dict
        pass

    @abstractmethod
    def play(self, round):
        # Play a round with a given context
        # Returns list of arms
        # Advise shuffling the plays before returning, don't want to bias adding then deleting nodes for instance
        pass

    @abstractmethod
    def update(self, arm, plays, reward):
        # Update arm by arm with a given reward and adjustable play amount
        # If multiple arms are played, adjust each arm's reward and plays externally (i.e. in reproduction)
        pass

    @abstractmethod
    def report(self):
        # Return a report about itself
        pass

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
            self.arms = [0 for _ in range(6)]
            self.plays = [0 for _ in range(6)]

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

        if "n_plays" in params:
            self.n = params["n_plays"]
        else:
            self.n = [1]

        if "sf_0" in params:
            for arm in params["sf_0"]:
                self.arms.append(arm)
                self.plays.append(0)

        else:
            for _ in range(6):
                self.arms.append((0,0))
                self.plays.append(0)

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

        score = max(rewards)
        if score > 0:
            s += score
        else:
            f -= score

        self.arms[arm] = (s,f)
        self.plays[arm] += plays

    def report(self):
        for i in range(len(self.arms)):
            print("{}: beta:{}, plays:{}".format(i, self.arms[i], self.plays[i]))


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