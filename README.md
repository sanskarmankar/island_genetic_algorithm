# Island Genetic Algorithm (island_ga)

A fast, parallel, and reusable Island Genetic Algorithm library for optimizing parameter combinations in quantitative trading strategies, machine learning models, or any custom optimization problem.

This implementation supports:

* Parallel fitness evaluation
* Multi-island evolution
* Migration between islands
* Tournament selection
* Mutation and crossover
* Fully customizable fitness function
* Clean integration with Pandas DataFrames or any custom context

---

# Installation

## Install from GitHub

```bash
pip install git+https://github.com/YOUR_USERNAME/island_ga.git
```

## Install locally (development mode)

```bash
git clone https://github.com/YOUR_USERNAME/island_ga.git
cd island_ga
pip install -e .
```

---

# Core Concept

The algorithm evolves parameter combinations to maximize a user-defined fitness function.

You provide:

* Parameter search space (`param_lists`)
* Fitness function (`fitness_function`)
* Optional context (DataFrame, model, etc.)

The library handles:

* Population creation
* Selection
* Crossover
* Mutation
* Parallel evaluation
* Migration between islands

---

# Required Inputs

## 1. param_lists (REQUIRED)

Defines the parameter search space.

Type:

```
List[List[Any]]
```

Example:

```python
param_lists = [
    [36, 48, 60, 72],       # norm lookback
    [0.8, 0.9, 0.95],       # quantile threshold
    [15, 20, 25],           # ADX strength
    [0.5, 1.0, 1.5],        # kappa
]
```

Each inner list represents possible values for one parameter.

---

## 2. fitness_function (REQUIRED)

This function evaluates one individual.

Signature:

```python
def fitness_function(individual, context):
    return fitness_value
```

Parameters:

| Parameter  | Description                                     |
| ---------- | ----------------------------------------------- |
| individual | List of parameter values                        |
| context    | Optional external data (DataFrame, model, etc.) |

Return:

```
float
```

Higher value = better solution

Example:

```python
def fitness_function(ind, df):

    norm_lb, quantile, strength, kappa = ind

    # Run your trading strategy here
    profit = run_strategy(df, norm_lb, quantile, strength, kappa)

    return profit
```

---

## 3. context (OPTIONAL but recommended)

External object passed to fitness function.

Usually:

* Pandas DataFrame
* Historical price data
* ML dataset
* Model reference

Example:

```python
context = df
```

---

# Optional Configuration Parameters

These control GA behavior.

```python
IslandGeneticAlgorithm(
    param_lists,
    fitness_function,
    context=None,
    islands=5,
    pop_size=20,
    generations=50,
    migration_interval=5,
    mutation_rate=0.7,
    n_jobs=4
)
```

Explanation:

| Parameter          | Description                    | Recommended |
| ------------------ | ------------------------------ | ----------- |
| islands            | Number of parallel populations | 4–16        |
| pop_size           | Population per island          | 20–100      |
| generations        | Evolution iterations           | 50–500      |
| migration_interval | Migration frequency            | 3–10        |
| mutation_rate      | Mutation probability           | 0.5–0.9     |
| n_jobs             | Parallel workers               | CPU cores   |

---

# Expected Output

The `.run()` method returns:

```python
best_params, best_fitness
```

Example:

```python
best_params, best_fitness = ga.run()

print(best_params)
print(best_fitness)
```

Output example:

```python
[60, 0.9, 25, 1.0]
2.45
```

Meaning:

```
Best parameter combination = [60, 0.9, 25, 1.0]
Best fitness score = 2.45
```

---

# Full Usage Example

```python
import pandas as pd
from island_ga import IslandGeneticAlgorithm

# Load data
df = pd.read_csv("data.csv")

# Define parameter space
param_lists = [
    [36,48,60],
    [0.8,0.9],
    [15,20,25],
]

# Define fitness function
def fitness(ind, df):

    norm_lb, quantile, strength = ind

    profit = (
        norm_lb * 0.01 +
        quantile * 2 +
        strength * 0.05
    )

    return profit

# Create GA object
ga = IslandGeneticAlgorithm(
    param_lists=param_lists,
    fitness_function=fitness,
    context=df,
    islands=5,
    pop_size=20,
    generations=10,
    migration_interval=3,
    n_jobs=4
)

# Run optimization
best_params, best_fitness = ga.run()

print("Best Parameters:", best_params)
print("Best Fitness:", best_fitness)
```

---

# How the Algorithm Works

Step 1: Initialization

Creates multiple islands with random populations.

Step 2: Fitness Evaluation

Evaluates each individual using your fitness function.

Step 3: Selection

Uses tournament selection to choose best individuals.

Step 4: Crossover

Combines parent genes to create offspring.

Step 5: Mutation

Randomly changes parameter values to explore new solutions.

Step 6: Migration

Best individuals migrate between islands to prevent local optima.

Step 7: Repeat

Process repeats for specified generations.

Step 8: Final Output

Returns best parameters and fitness score.

---

# Parallel Processing

Fitness evaluation runs in parallel using ThreadPoolExecutor.

This significantly speeds up optimization for heavy fitness functions such as:

* Trading strategy backtests
* Machine learning training
* Simulation models

---

# Use Cases

This library is ideal for optimizing:

* Trading strategies
* Machine learning hyperparameters
* Signal processing parameters
* Portfolio optimization
* Any custom objective function

---

# Performance

Typical speed improvement vs brute force:

```
10x to 1000x faster
```

Especially useful for large parameter spaces.

---



# Author

Sanskar Mankar
