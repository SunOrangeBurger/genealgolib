# GeneAlgoLib

A blazingly fast genetic algorithm library implemented in Rust with Python bindings via PyO3.

## Features

- **High Performance**: Rust implementation provides significant speedup over pure Python/NumPy implementations
- **Simple API**: Easy-to-use Python interface
- **Tournament Selection**: Efficient selection mechanism with configurable tournament size
- **Elitism**: Preserves best individuals across generations
- **Configurable**: Adjustable population size, mutation rate, and gene count

## Installation

```bash
pip install genealgolib
```

## Quick Start

```python
import numpy as np
from fast_ga import GeneticAlgorithm

# Initialize GA
ga = GeneticAlgorithm(
    pop_size=1000,      # Population size
    num_genes=10,       # Number of genes per individual
    mutation_rate=0.05  # Mutation probability
)

# Evolution loop
for generation in range(100):
    # Get current population
    population = ga.get_population()
    
    # Calculate fitness scores (example: sum to 10.0)
    sums = np.sum(population, axis=1)
    fitness_scores = 1.0 / (np.abs(10.0 - sums) + 1e-6)
    
    # Evolve to next generation
    ga.evolve(fitness_scores)

# Get final population
final_pop = ga.get_population()
```

## Performance

GeneAlgoLib significantly outperforms highly optimized NumPy implementations:

- **Average Speedup**: 2-5x faster than vectorized NumPy
- **Scales Well**: Better performance with larger populations
- **Memory Efficient**: Optimized memory usage in Rust

## API Reference

### `GeneticAlgorithm(pop_size, num_genes, mutation_rate)`

Initialize a genetic algorithm instance.

**Parameters:**
- `pop_size` (int): Number of individuals in the population
- `num_genes` (int): Number of genes per individual
- `mutation_rate` (float): Probability of mutation (0.0 to 1.0)

### `get_population()`

Returns the current population as a NumPy array of shape `(pop_size, num_genes)`.

### `evolve(fitness_scores)`

Evolve the population to the next generation.

**Parameters:**
- `fitness_scores` (np.ndarray): 1D array of fitness scores for each individual

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
