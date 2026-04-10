use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1}; // FIXED: Changed rust_numpy to numpy
use rand::Rng; // FIXED: Removed unused SliceRandom

#[pyclass]
pub struct GeneticAlgorithm {
    pop_size: usize,
    num_genes: usize,
    mutation_rate: f64,
    population: Vec<f64>,
}

#[pymethods]
impl GeneticAlgorithm {
    #[new]
    fn new(pop_size: usize, num_genes: usize, mutation_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let total_genes = pop_size * num_genes;
        
        let population: Vec<f64> = (0..total_genes)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        GeneticAlgorithm { pop_size, num_genes, mutation_rate, population }
    }

    fn get_population<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        // Create a 1D array, reshape it to 2D, and return it to Python
        let array = PyArray1::from_vec(py, self.population.clone());
        let array2d = array.reshape([self.pop_size, self.num_genes]).unwrap();
        array2d.into()
    }

    fn evolve(&mut self, fitness_scores: PyReadonlyArray1<f64>) {
        let fitness = fitness_scores.as_slice().unwrap();
        let mut rng = rand::thread_rng();
        let mut new_population = Vec::with_capacity(self.pop_size * self.num_genes);

        // Elitism: Keep the best individual
        let mut best_idx = 0;
        let mut best_fitness = f64::MIN;
        for (i, &f) in fitness.iter().enumerate() {
            if f > best_fitness {
                best_fitness = f;
                best_idx = i;
            }
        }

        let best_start = best_idx * self.num_genes;
        let best_end = best_start + self.num_genes;
        new_population.extend_from_slice(&self.population[best_start..best_end]);

        // Generate the rest of the population
        for _ in 1..self.pop_size {
            let parent1_idx = tournament_selection(fitness, 3, &mut rng);
            let parent2_idx = tournament_selection(fitness, 3, &mut rng);

            let p1_start = parent1_idx * self.num_genes;
            let p2_start = parent2_idx * self.num_genes;

            for gene_idx in 0..self.num_genes {
                let mut gene = if rng.gen_bool(0.5) {
                    self.population[p1_start + gene_idx]
                } else {
                    self.population[p2_start + gene_idx]
                };

                if rng.gen::<f64>() < self.mutation_rate {
                    gene += rng.gen_range(-0.5..0.5);
                }
                new_population.push(gene);
            }
        }
        self.population = new_population;
    }
}

fn tournament_selection(fitness: &[f64], k: usize, rng: &mut impl Rng) -> usize {
    let mut best_idx = rng.gen_range(0..fitness.len());
    let mut best_fit = fitness[best_idx];

    for _ in 1..k {
        let idx = rng.gen_range(0..fitness.len());
        if fitness[idx] > best_fit {
            best_fit = fitness[idx];
            best_idx = idx;
        }
    }
    best_idx
}

// FIXED: Updated signature to fix the deprecation warning
#[pymodule]
fn fast_ga(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GeneticAlgorithm>()?;
    Ok(())
}