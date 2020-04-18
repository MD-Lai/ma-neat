from neat.reporting import BaseReporter
from copy import deepcopy

class EarlyTerminationReporter(BaseReporter):
    def __init__(self):
        self.early = False
        self.generation = 0
        self.winner = None

class BanditReporter(BaseReporter):
    def __init__(self):
        self.generational_data = []
    def post_reproduction(self, cfg, bandit, ignore):
        self.generational_data.append(deepcopy(bandit.get_data()))
        
class BanditPrintReporter(BanditReporter):

    def post_reproduction(self, cfg, bandit, ignore):
        print(bandit.report())
        super().post_reproduction(cfg, bandit, ignore)