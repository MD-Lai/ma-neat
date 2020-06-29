# ma-neat

Incorporating Multi-Armed Bandits into NEAT

# WHY

Because I hypothesised that some mutations would be more likely to benefit the performance of a network than others.
The bandit controls which mutations are applied and observes the reward. 
The reward part is a ratio of positive to negative rewards. 

Masters project for Computer Science 

# How to Use
Test Environments and Bandit options are allocated in main_mp2.py. Processing files are interfaced through process_pipeline.py

## run 
Running tests: python3 main_mp2.py {test\_id} {bandit\_id} {n\_cpus} {lowest\_index}

test_ids: (int) \[0,7\]

bandit_ids: (int) \[0,17\]

n_cpus: (int) how many cpus you want to use

lowest_index: (int) each cpu runs a tests, tests are saved according to cpu_id (\[0,n\_cpus-1\]) + lowest_index, so tests can be rerun multiple times with increasing indeces and all results are saved in separate files. 

## process
Processing data: python3 process_pipeline.py

Make sure results are placed in folder named spartan/test\_{test_id}\_{bandit_id} for it to work (sorry for the hard coding I didn't have the time to fix it up and things stuck too hard and changing would require an entire rewrite)

processes test 0,1,4,5,7 with bandits \[0,17\] assuming 32 runs (n\_cpus) each

produces a box plot, dominance file (statistical comparisons),  fitness scatter, mean performances, and full pairwise comparisons. 

That's all, I may add more to readme at a later stage.
