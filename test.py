import numpy as np
import time
from fast_ga import GeneticAlgorithm

POP_SIZE = 10_000
NUM_GENES = 5
MUTATION_RATE = 0.05
EPOCHS = 100

print("Initializing Rust Genetic Algorithm...")
ga = GeneticAlgorithm(POP_SIZE, NUM_GENES, MUTATION_RATE)

start_time = time.time()

for epoch in range(EPOCHS):
    # 1. Get Population (Zero-copy from Rust)
    pop = ga.get_population()
    
    # 2. Evaluate Fitness (in Python using NumPy)
    sums = np.sum(pop, axis=1)
    distance = np.abs(10.0 - sums)
    fitness_scores = 1.0 / (distance + 1e-6) 
    
    # 3. Evolve next generation (in Rust)
    ga.evolve(fitness_scores)
    
    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        best_sum = sums[np.argmax(fitness_scores)]
        print(f"Epoch {epoch:3} | Best Sum: {best_sum:.4f} | Target: 10.0")

print(f"\nEvolution finished in {time.time() - start_time:.4f} seconds!")