import pygad
import numpy
function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44
omax = 0
omin = 0

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness

def on_start(ga_instance):
    global omax
    global omin
    print(f"Generation = {ga_instance.generations_completed}")

    for idx in range(ga_instance.population.shape[0]):
        solution = ga_instance.population[idx]
        print(f"index = {idx} Solution = {solution}")
        print(type(solution))
def on_generation(ga_instance):

    print(f"Generation = {ga_instance.generations_completed}")

    for idx in range(ga_instance.population.shape[0]):
        solution = ga_instance.population[idx]
        fitness = ga_instance.last_generation_fitness[idx]
        print(f"index = {idx} Solution = {solution}")
        print(type(solution))
fitness_function = fitness_func

num_generations = 4
num_parents_mating = 4

sol_per_pop = 8
num_genes = len(function_inputs)

init_range_low = -2
init_range_high = 5

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       on_start = on_start,
                       on_generation = on_generation)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = numpy.sum(numpy.array(function_inputs)*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))