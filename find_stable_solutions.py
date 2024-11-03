import time
from datetime import datetime

from cli_simlator import *


def evaluate_solution(solution):
    global planet_masses, planet_positions, planet_velocities
    planet_masses = np.random.uniform(1e20, 1e30, num_planets)  # 随机生成星体的质量
    planet_positions = solution  # 使用 solution 作为行星的位置
    planet_velocities = np.random.uniform(
        -1e4, 1e4, (num_planets, 3)
    )  # 随机生成星体的速度
    history = {"planet": [], "time": [], "position": []}
    for data in simlator():
        history["planet"].append(data["planet"])
        history["time"].append(data["time"])
        history["position"].append(data["pos"])
    planet_indices = range(len(planet_masses))  # 所有行星的索引
    regularities = [
        check_regularity(history, planet_index) for planet_index in planet_indices
    ]
    return all(regularities)


@jit(nopython=NOPYTHON, fastmath=FASTMATH)
def crossover(parent1, parent2):
    child1 = np.copy(parent1)
    child2 = np.copy(parent2)
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1[crossover_point:] = parent2[crossover_point:]
    child2[crossover_point:] = parent1[crossover_point:]
    return child1, child2


@jit(nopython=NOPYTHON, fastmath=FASTMATH)
def mutate(solution, mutation_rate=0.01):
    for i in range(len(solution)):
        if np.random.rand() < mutation_rate:
            solution[i] = np.random.uniform(-1e12, 1e12)
    return solution


# @jit(nopython=NOPYTHON)
def genetic_algorithm(num_generations=100, population_size=10, mutation_rate=0.01):
    population = [
        np.random.uniform(-1e12, 1e12, (num_planets, 3)) for _ in range(population_size)
    ]
    for _ in range(num_generations):
        fitness_scores = [evaluate_solution(solution) for solution in population]

        # 根据适应度分数对种群进行排序
        sorted_population = sorted(
            zip(population, fitness_scores), key=lambda x: x[1], reverse=True
        )

        # 提取排序后的种群和适应度分数
        population = [sol for sol, _ in sorted_population]
        fitness_scores = [fit for _, fit in sorted_population]

        if population[0].size > 0 and population[0].all():
            return population[0]
        new_population = [population[0]]  # 保留最佳解
        for _ in range(population_size - 1):
            parent1 = np.random.choice(population[:5])  # 选择前5个最佳解作为父代
            parent2 = np.random.choice(population[:5])
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        population = new_population
    return population[0]


def find_global_optimum(num_queries=30):
    best_solution = None
    best_fitness = -1
    for iter_count in range(num_queries):
        solution = genetic_algorithm()
        fitness = evaluate_solution(solution)
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = solution
        print(f"Finished iter: {iter_count}", end="\r")
    return best_solution


if __name__ == "__main__":
    initial_condition = {
        "G": G,
        "dt": dt,
        "total_time": total_time,
        "num_planets": num_planets,
        "planet_masses": planet_masses,
        "planet_positions": planet_positions,
        "planet_velocities": planet_positions,
    }

    seconds = total_time
    years = seconds // (365 * 24 * 3600)
    seconds %= 365 * 24 * 3600
    months = seconds // (30 * 24 * 3600)
    seconds %= 30 * 24 * 3600
    days = seconds // (24 * 3600)
    seconds %= 24 * 3600
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    print(
        f"Initial condition: {initial_condition}\nBegin at(RealTime): {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}\nTotal simlated time(SimlatorTime): {f'{years}-{months}-{days} {hours}:{minutes}:{seconds}'}"
    )
    solution = find_global_optimum()
    print(
        f"Stable solution found: {solution}\nEnd at(RealTime): {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}"
    )
