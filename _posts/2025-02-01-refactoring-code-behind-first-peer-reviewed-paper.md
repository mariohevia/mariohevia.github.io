---
layout: distill
# title: "Seven Years Later: Refactoring the Code Behind My Master's Thesis"
title: "Seven Years Later: Refactoring the Code Behind My First Peer-Reviewed Paper"
date: 2025-02-01
description: I revisit the code from my first peer-reviewed paper (and master's thesis), published seven years ago, to modernize it with the skills and knowledge I’ve gained since, improving its performance, readability, and maintainability.
tags: Python Jax Evolutionary-Computation GPU
categories: 
authors:
  - name: Mario A. Hevia Fajardo
    url: "https://mhevia.com"
    affiliations:
      name: University of Birmingham

toc:
  - name: What does the original code do?
  - name: What I did right?
  - name: Improvements
  - name: Is the code's behavior preserved?
  - name: Runtime comparisons
  - name: Conclusions
---

After seven years of programming and learning, I decided to revisit one of my first major Python projects, because I want to reuse it for a new project. Although I had been programming in Python for over a year at the time, I was still relatively new to it and many things can be improved in the codebase. Therefore, my plan is to refactor and modernise the code in order to be able to reporpose it and at the same time improve its runtime using GPUs.

In this post, I'll share what I think I did right, and what steps I took to improve it.

### What does the original code do?

This project was part of my Master's Thesis and eventually contributed to my [first peer-reviewed publication](https://doi.org/10.1145/3321707.3321858) at [GECCO 2019](https://gecco-2019.sigevo.org/index.html/HomePage), the ACM flagship conference in evolutionary computation. 
In the project I aimed to compare previously published self-adjusting evolutionary algorithms (EAs) and improve upon them. If you are unfamiliar with self-adjusting EAs, don't worry, I will explain it in simple terms. 

EAs are optimisation algorithms that mimic evolution to search for a solution to a problem. They work similar to trial and error: you start with a set of solutions, then randomly modify (mutate) or combine (crossover) them. Next, you update your current set of solutions based on how well they perform, compared to the initial solutions, finally, you repeat the process until you find a good solution. While the idea is simple, there are many different EAs, and each comes with parameters like the number of solutions created every iteration (offspring population) or how much the new solutions are changed (mutation and crossover rate), and the parameter settings can greatly affect their performance.

When using an EA, practitioners can either select the parameters at the start of the run (which remain fixed throughout) or let the algorithm adjust them on its own. Self-adjusting EAs fall into the latter category—they automatically choose their parameters without human input. If you're interested in this topic, you can read more in my [PhD thesis](self-adjusting).

That said, the code implements several well-known and some new self-adjusting EAs and compares their optimisation times (calculated by the number of function evaluations) on various optimisation problems.

### What I did right?

I think that, overall, the code is pretty good. It is well written, and I even did documentation for it, which helped this time around. 

Here are the things I think I did right:

- The algorithms and the optimisation problems are implemented as classes with a consistent structure, making them interchangable and allowing new algorithms/problems to be added without changing the main file.
- The project is relatively well-structured. Here is the file structure:
```python
code/ 
├── master.py           # Main script
├── utils/ 
    ├── inputs.py         # Parsing command-line parameters
    ├── leadingones.py    # LeadingOnes problem
...
    ├── outputs.py        # Plots and output logs
    └── sufsamp.py        # SufSamp problem
```
- Each algorithm and problem is contained in its own file, making it easy to maintain.
- The main file accepts command-line parameters, providing flexibility for running different experiments.
- All parameters are validated, with explanations and suggested values provided for invalid inputs.
- Experiments are logged and plotted automatically after completion.<!-- , although only the aggregated values of several runs is logged and not every run. -->


### Improvements

Although the project has a good structure, there are still many things that can be improved or modernised with new libraries. Let's go through them by parts.

#### Code structure, modularity and reusability

The current implementation uses two "types of classes", but there is no template class for them to inherit from. Because of this, the code assumes that certain class methods exist and that specific attributes are created and updated inside the algorithm and problem classes without prior checks. It also directly accesses these attributes outside of the class, which is not good practice.

Additionally, the algorithms assume that the representations of the problem's solutions are strings consisting only of ones and zeroes, which limits the possible representations of a problem. This is done because the mutation and crossover used by the algorithms depend on the problem representation, but this should not be set in stone. A better approach is to let the problem class handle the representation of the problems, including how they can be mutated or combined.

Let's write the two parent classes:

```python
class Problem:
    def evaluate(self):
        raise NotImplementedError("Subclass must implement evaluate method")

    def get_name(self):
        raise NotImplementedError("Subclass must implement get_name method")

    def get_max_fitness(self):
        raise NotImplementedError("Subclass must implement get_max_fitness method")

    def sample_uar(self):
        raise NotImplementedError("Subclass must implement mutate method")

    def to_string(self):
        raise NotImplementedError("Subclass must implement to_string method")

    def get_shape(self):
        raise NotImplementedError("Subclass must implement to_string method")

    def mutate(self):
        raise NotImplementedError("Subclass must implement mutate method")

    def crossover(self):
        raise NotImplementedError("Subclass must implement crossover method")

    def multiparent_crossover(self):
        raise NotImplementedError("Subclass must implement multiparent_crossover method")
```
```python
class Algorithm:
    def __init__(self, problem):
        self.num_evaluations = 0

    def __next__(self):
        raise NotImplementedError("Subclass must implement __next__ method")

    def get_num_evaluations(self):
        return self.num_evaluations

    def get_best_individual(self):
        raise NotImplementedError("Subclass must implement get_best_individual method")

    def get_current_population(self):
        raise NotImplementedError("Subclass must implement get_best_individual method")
```

Since the current implementation only deals with pseudo-Boolean problems (bitstring representations), I also implemented a new subclass that handles the mutation and crossover for all these problems. This way, the final classes only include the necessary functions, avoiding boilerplate code. As an example, here I show OneMax, which assigns fitness based on the number of one-bits in the bitstring.
```python
class OneMax(BitstringProblem):
    def evaluate(self, bitstring, _):
        return jnp.sum(bitstring)
    
    def get_name(self):
        return "OneMax"

    def get_max_fitness(self):
        return self.n
```
Once all the problems were implemented, I returned to the algorithms. However, for this part, we need to consider possible optimisations.

#### Parallelisation and optimisations

One of the main goals of revisiting this project is to improve the script's runtime. Since I first considered refactoring the code, I also planned to enable it to run on a GPU using JAX, which should improve the runtime on its own. However, we also need to identify the parts of the code that take the most time during execution to prioritise optimising these areas.

To do that I will use the line-by-line profiling library [line_profiler](https://github.com/pyutils/line_profiler). I installed it with:

```bash
pip install line-profiler
```

Once installed, I added the decorator <code class="language-python">@profile</code> to the main function in <code>master.py</code> and then run:

```bash
kernprof -l -v master.py
```
This highlighted that, intuitively, the line that takes the most time is <code class="language-python">next(algorithm)</code>. This is where the script iterates the algorithm.

```python
Line    Hits         Time  Per Hit   % Time  Line Contents
==============================================================
...
75       100         37.3      0.4      0.0          if configuration.stop_criteria == 'solved':
76       100        369.3      3.7      0.0              m_size = max(100 * math.pow(configuration.problem_size,4), 1000000)
77    106957      18801.5      0.2      0.2              while not algorithm.solved:
78    106857    9996295.9     93.5     97.1                  next(algorithm)
79    106857      23680.5      0.2      0.2                  if (algorithm.parent[0][0] == past_fitness):
80    102403      14566.7      0.1      0.1                      tol += 1
81    102403      15213.8      0.1      0.1                      tol2 += algorithm.offspring_size
82                                                           else:
83      4454        577.7      0.1      0.0                      tol = 0
84      4454        506.4      0.1      0.0                      tol2 = 0
85    106857      16685.3      0.2      0.2                  past_fitness = algorithm.parent[0][0]
86    106857      34697.3      0.3      0.3                  if (algorithm.evaluations > m_size or tol2 > 1000000):
...
```

Although each algorithm performs different tasks, I profiled one of the simplest algorithms to get an idea of which processes take the longest. From the results below, we can see that creating new offspring through mutation and selecting the best offspring take the most time. Therefore, I will focus on these areas when rewriting the code.


```python
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
...
30                                               def __next__(self):
31    108166      18012.9      0.2      0.1          offspring = []
32    216332     139315.8      0.6      0.8          self.parent = [self.problem.evaluate(self.bit_string), 
33    108166      14133.1      0.1      0.1                          self.bit_string]
34    108166      19152.2      0.2      0.1          self.mut_prob_gen.append(self.mutation_probability)
35    108166      16763.9      0.2      0.1          self.lambda_gen.append(self.offspring_size)
36    216332      63976.3      0.3      0.4          for i in range(self.offspring_size):
37    108166   13799267.8    127.6     82.0              mutated_string = self.mutate()
38    216332     132957.3      0.6      0.8              offspring.append((self.problem.evaluate(mutated_string), 
39    108166      13945.3      0.1      0.1                                mutated_string))
40    108166    2534601.8     23.4     15.1          self.select(offspring)
41    108166      22606.2      0.2      0.1          self.generations += 1
42    108166      45919.3      0.4      0.3          self.evaluations += self.offspring_size
```

The good news from these results is that the tasks taking the most time can be improved through parallelisation. I will now show you how the same function looks in the new code. You’ll notice that I added more comments to make the code more maintainable, and I parallelised the tasks that take the longest.

```python
    def __next__(self):
        # Split RNG key for mutation and generate subkeys for each offspring mutation
        self.rng_key, mutate_subkey = jax.random.split(self.rng_key)
        mutate_subkeys = jax.random.split(mutate_subkey, self.offspring_size)

        # Generate offspring by mutating the parent using parallelisation
        vmap_mutation = jax.vmap(self.problem.mutate, in_axes=(None, None, 0))
        offspring = vmap_mutation(self.parent,self.mutation_probability,mutate_subkeys)

        # Combine offspring with the original parent for evaluation
        offspring_parent = jnp.concatenate([offspring,self.parent[jnp.newaxis, :]], axis=0)

        # Split RNG key for evaluation and generate subkeys for each evaluation
        self.rng_key, eval_subkey = jax.random.split(self.rng_key)
        eval_subkeys = jax.random.split(eval_subkey, self.offspring_size+1)

         # Evaluate fitness of offspring and parent
        vmap_eval = jax.vmap(self.problem.evaluate)
        fitnesses = vmap_eval(offspring_parent,eval_subkeys)

        # Identify the index of the best fitness
        best_index = jnp.argmax(fitnesses)

        # Update the parent, best fitness and number of evaluations used
        self.parent = offspring_parent[best_index]
        self.num_evaluations += self.offspring_size
        self.best_fitness = max(self.best_fitness, fitnesses[best_index])
```


### Is the code's behavior preserved?

Before we start comparing the runtime of both implementations, we need to ensure that the behaviour of the code remains consistent. Since the algorithms are random, the new implementation uses different libraries (with distinct random generators), and I have rewritten several parts of the code, we can’t expect the exact same results, even with the same random seeds. However, we can still test whether the results are consistent on average by running multiple tests.

Testing with the same algorithm I showed earlier on OneMax for 100 runs with both implementations, I obtained an average of 124,905 function evaluations with the old code and 125,225 function evaluations with the new one. This suggests that the two are likely to have the same behavior and we can continue to comparing their runtime.

### Runtime comparisons

The final part is to compare the runtime of the new implementation versus the old one. For these tests, I used the same algorithm-problem pair as before, and we can clearly see a stark difference in performance. One finishes in just over one and a half minutes, while the other takes more than an hour!

Here are the results (I've removed most of the lines that didn’t take too much time):

##### New implementation
```python
Total time: 101.373 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
...
    37       100         35.3      0.4      0.0              case "solved":
    38     25145    2513597.3    100.0      2.5                  while algorithm.best_fitness<problem.get_max_fitness():
    39     25045   97746846.7   3902.8     96.4                      next(algorithm)
```

##### Old implementation
```python
Total time: 101.373 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
...
    37       100         35.3      0.4      0.0              case "solved":
    38     25145    2513597.3    100.0      2.5                  while algorithm.best_fitness<problem.get_max_fitness():
    39     25045   97746846.7   3902.8     96.4                      next(algorithm)
```

### Conclusion

The first thing I want to say is that it was so much fun to return to one of my first projects and realise how much I've learnt over the last 7 years. The results show that the new implementation not only preserves the behaviour of the original code but also runs much faster, making it more efficient for larger experiments. Now, I can move on to my next project, which will use this new implementation as a base.