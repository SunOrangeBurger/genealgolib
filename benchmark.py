import numpy as np
import time
from fast_ga import GeneticAlgorithm as RustGA

# ==========================================
# 1. THE PURE PYTHON / NUMPY IMPLEMENTATION
# ==========================================
class NumpyGA:
    def __init__(self, pop_size, num_genes, mutation_rate):
        self.pop_size = pop_size
        self.num_genes = num_genes
        self.mutation_rate = mutation_rate
        # Initialize population
        self.population = np.random.uniform(-1.0, 1.0, (pop_size, num_genes))

    def get_population(self):
        return self.population

    def evolve(self, fitness_scores):
        new_population = np.empty_like(self.population)
        
        # 1. Elitism: Keep the best individual unconditionally
        best_idx = np.argmax(fitness_scores)
        new_population[0] = self.population[best_idx]
        
        # 2. Tournament Selection
        # We need 2 parents for every child (pop_size - 1 children)
        num_parents = 2 * (self.pop_size - 1)
        
        # Pick 3 random contenders for each parent spot
        contenders = np.random.randint(0, self.pop_size, (num_parents, 3))
        contender_fitness = fitness_scores[contenders]
        
        # Find the winner of each tournament
        winners_idx = np.argmax(contender_fitness, axis=1)
        parents = contenders[np.arange(num_parents), winners_idx]
        
        # Split into parent 1 and parent 2 arrays
        p1_indices = parents[0::2]
        p2_indices = parents[1::2]
        
        # 3. Crossover (50% chance from either parent)
        crossover_mask = np.random.rand(self.pop_size - 1, self.num_genes) > 0.5
        children = np.where(crossover_mask, self.population[p1_indices], self.population[p2_indices])
        
        # 4. Mutation
        mutation_mask = np.random.rand(self.pop_size - 1, self.num_genes) < self.mutation_rate
        mutation_noise = np.random.uniform(-0.5, 0.5, (self.pop_size - 1, self.num_genes))
        children += mutation_mask * mutation_noise
        
        # Assign children to the new generation
        new_population[1:] = children
        self.population = new_population


# ==========================================
# 2. THE BENCHMARK FUNCTION
# ==========================================
def run_benchmark(ga_engine, name, epochs=200):
    print(f"--- Running {name} ---")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Get Population
        pop = ga_engine.get_population()
        
        # Evaluate Fitness (Summing to 10.0)
        sums = np.sum(pop, axis=1)
        distance = np.abs(10.0 - sums)
        fitness_scores = 1.0 / (distance + 1e-6) 
        
        # Evolve next generation
        ga_engine.evolve(fitness_scores)
        
    end_time = time.time()
    
    # Get final best
    pop = ga_engine.get_population()
    best_sum = np.sum(pop[np.argmax(fitness_scores)])
    duration = end_time - start_time
    
    print(f"Time Taken: {duration:.4f} seconds")
    print(f"Best Sum:   {best_sum:.4f}")
    print("-" * 25 + "\n")
    return duration


if __name__ == "__main__":
    # Test configurations
    POPULATION_SIZES = [1_000, 5_000, 10_000, 20_000]
    GENERATION_COUNTS = [50, 100, 200, 500]
    MUTATION_RATES = [0.01, 0.05, 0.1, 0.2]
    NUM_GENES = 10  # Keep constant for consistency
    
    results = []
    
    print("=" * 80)
    print("COMPREHENSIVE GENETIC ALGORITHM BENCHMARK")
    print("=" * 80)
    print(f"Testing {len(POPULATION_SIZES)} population sizes × {len(GENERATION_COUNTS)} generation counts × {len(MUTATION_RATES)} mutation rates")
    print(f"Total configurations: {len(POPULATION_SIZES) * len(GENERATION_COUNTS) * len(MUTATION_RATES)}")
    print("=" * 80)
    print()
    
    config_num = 0
    total_configs = len(POPULATION_SIZES) * len(GENERATION_COUNTS) * len(MUTATION_RATES)
    
    for pop_size in POPULATION_SIZES:
        for generations in GENERATION_COUNTS:
            for mutation_rate in MUTATION_RATES:
                config_num += 1
                print(f"\n{'='*80}")
                print(f"Configuration {config_num}/{total_configs}")
                print(f"Population: {pop_size:,} | Generations: {generations} | Mutation Rate: {mutation_rate}")
                print(f"{'='*80}")
                
                # Run NumPy GA
                numpy_ga = NumpyGA(pop_size, NUM_GENES, mutation_rate)
                numpy_time = run_benchmark(numpy_ga, "NumPy GA", generations)
                
                # Run Rust GA
                rust_ga = RustGA(pop_size, NUM_GENES, mutation_rate)
                rust_time = run_benchmark(rust_ga, "Rust GA", generations)
                
                # Calculate speedup
                speedup = numpy_time / rust_time
                
                # Store results
                results.append({
                    'pop_size': pop_size,
                    'generations': generations,
                    'mutation_rate': mutation_rate,
                    'numpy_time': numpy_time,
                    'rust_time': rust_time,
                    'speedup': speedup
                })
                
                print(f"⚡ Speedup: {speedup:.2f}x (Rust is {speedup:.2f}x faster)")
    
    # Summary Report
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"\n{'Pop Size':<12} {'Gens':<8} {'Mut Rate':<12} {'NumPy (s)':<12} {'Rust (s)':<12} {'Speedup':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['pop_size']:<12,} {r['generations']:<8} {r['mutation_rate']:<12.2f} "
              f"{r['numpy_time']:<12.4f} {r['rust_time']:<12.4f} {r['speedup']:<10.2f}x")
    
    # Statistics
    speedups = [r['speedup'] for r in results]
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Average Speedup:  {np.mean(speedups):.2f}x")
    print(f"Median Speedup:   {np.median(speedups):.2f}x")
    print(f"Min Speedup:      {np.min(speedups):.2f}x")
    print(f"Max Speedup:      {np.max(speedups):.2f}x")
    print(f"Std Dev:          {np.std(speedups):.2f}x")
    
    # Best and worst cases
    best_case = max(results, key=lambda x: x['speedup'])
    worst_case = min(results, key=lambda x: x['speedup'])
    
    print("\n" + "=" * 80)
    print("BEST CASE (Highest Speedup)")
    print("=" * 80)
    print(f"Population: {best_case['pop_size']:,} | Generations: {best_case['generations']} | Mutation: {best_case['mutation_rate']}")
    print(f"Speedup: {best_case['speedup']:.2f}x")
    
    print("\n" + "=" * 80)
    print("WORST CASE (Lowest Speedup)")
    print("=" * 80)
    print(f"Population: {worst_case['pop_size']:,} | Generations: {worst_case['generations']} | Mutation: {worst_case['mutation_rate']}")
    print(f"Speedup: {worst_case['speedup']:.2f}x")
    
    print("\n" + "=" * 80)
    print(f"🏆 Overall: Rust is {np.mean(speedups):.2f}x faster on average!")
    print("=" * 80)