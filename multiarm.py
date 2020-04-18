from neat import DefaultGenome, DefaultReproduction
from neat.genome import DefaultGenomeConfig

import math
from random import random, choice
from itertools import count
from statistics import median

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean
from neat.six_util import iteritems, itervalues
from neat.genes import DefaultConnectionGene, DefaultNodeGene

class BanditReproduction(DefaultReproduction):

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2)])
    def bandit_type(self, bandit):
        # Take a bandit from external sources
        self.bandit = bandit

    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        super().__init__(config, reporters, stagnation)

        self.records = [] # list of {id: (parent_fitness, mutation_directives)} per generation

        self.normalising = False

    # Seems like the right point to initiate a bandit, at the reproduction level
    # as the genome level is applied individually
    # Genome will use bandit as passed in by reproduction
    # Modified from the existing version
    # FIXME comments added to indicate work required to use bandit
    # most to be copied, but changes need to be injected to update bandit arms with prev fitness
    def reproduce(self, config, species, pop_size, generation):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """
        # TODO: I don't like this modification of the species and stagnation objects,
        # because it requires internal knowledge of the objects.

        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.
        all_fitnesses = []
        remaining_species = []

        old_population = []

        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            
            old_population.append(itervalues(stag_s.members))

            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in itervalues(stag_s.members))
                remaining_species.append(stag_s)
                

        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {} # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in itervalues(afs.members)])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses) # type: float
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size,self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)


        # Bandit Update
        # TODO Reconsider reward scheme to be invariant of actual fitness value
        # Percentage improvement to fitness?
        # Fitness improvement as a fraction of best improvement? - seems good, then one mutation is always 1, rest are < 1
        # How to reward or perceive "failure"? Same as fraction but for decreases? 
        # 1 if best arm in generation else 0?
        if generation > 0:
            mutation_delta = []

            for mutant, old_fitness, mutations in self.records[-1]:
                fit_delta = mutant.fitness - old_fitness
                
                for m in mutations:
                    mutation_delta.append((m, fit_delta))
            # print(mutation_delta)
            self.bandit.update_all(mutation_delta)
            #         if m not in mutations_deltas:
            #             mutations_deltas[m] = [fit_delta]
            #         else:
            #             mutations_deltas[m].append(fit_delta)

            # for mutations, deltas in mutations_deltas.items():
            #     self.bandit.update(mutations, deltas) # len deltas can be removed, the number of rewards is the number of plays
                
            # print([(g.key, g.fitness-f, m) for g,f,m in self.records[-1]])
        
        self.reporters.post_reproduction(config, self.bandit, None)


        self.records.append([])
        
        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(iteritems(s.members))
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *
                                         len(old_members)))
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                parent1_id, parent1 = choice(old_members)
                parent2_id, parent2 = (parent1_id, parent1)

                gid = next(self.genome_indexer)
                child = config.genome_type(gid)

                child.configure_crossover(parent1, parent2, config.genome_config)

                mutation_directives = self.bandit.play(generation)
                child.mutate(config.genome_config, mutation_directives)

                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id) 

                self.records[-1].append((child, parent1.fitness, mutation_directives))
        
        return new_population

class BanditGenome(DefaultGenome):
    
    def __init__(self, key):
        super().__init__(key)
        # self.last_mutations = None # TODO kept by reproduction
        # self.last_fitness = None # TODO kept by reproduction

    def mutate_mutate_nodes(self, config):
        for ng in self.nodes.values():
            ng.mutate(config)

    def mutate_mutate_connections(self,config):
        for cg in self.connections.values():
            cg.mutate(config)

    # bandit to be passed in for use rather than held by the genome 
    def mutate(self, config, mutation_directives):

        for m in mutation_directives:
            if m == 0:
                self.mutate_add_node(config)
                
            elif m == 1:
                self.mutate_delete_node(config)
                
            elif m == 2:
                self.mutate_mutate_nodes(config)
                
            elif m == 3:
                self.mutate_add_connection(config)
                
            elif m == 4:
                self.mutate_delete_connection()
                
            elif m == 5:
                self.mutate_mutate_connections(config)

            else:
                # I mean it shouldn't have gotten here but we'll see
                print("Nonononononononononononononononononononononononono")
                pass

        