import torch
import torch.nn as nn
from torchvision.models import vgg16
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class model:
    def __init__(self, prune_rate):
        # 定义VGG16模型
        self.model = vgg16(pretrained=False)
        self.model.classifier[6] = nn.Linear(4096, 100)  # 修改最后一层以适应CIFAR100

        # 加载训练好的模型参数
        self.model.load_state_dict(torch.load('vgg16_cifar100.pth', map_location=torch.device('cpu')))

        # 使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 获取全连接层的第一层参数矩阵
        fc1_weights = self.model.classifier[0].weight.data.cpu().numpy()

        # 将该矩阵分为16x98个256x256的方阵
        blocks = fc1_weights.reshape(16, 256, 98, 256).swapaxes(1, 2).reshape(16, 98, 256, 256)

        # 随机取出指定数量的块对并互换
        self.original_blocks = blocks
        self.select_idxs = []
        for _ in range(156):
            idx1 = (np.random.randint(16), np.random.randint(98))
            self.select_idxs.append(idx1)
            
    def valuate(self, permutation):
        
        new_blocks = self.original_blocks.copy()
        for i in range(1568):
            idx1 = (permutation[i]//98, permutation[i]%98)
            if permutation[i] in self.select_idxs:
                new_blocks[idx1] = self.original_blocks[idx1].copy()
            else:
                # zero
                new_blocks[idx1] = 0

        # 将修改后的参数矩阵重新赋值给模型
        fc1_weights = new_blocks.reshape(16, 98, 256, 256).swapaxes(1, 2).reshape(16, 256, 98, 256).reshape(4096, 25088)
        self.model.classifier[0].weight.data = torch.from_numpy(fc1_weights).to(self.device)

        # 测试模型
        self.model.eval()

        correct = 0
        total = 0

        # 加载CIFAR100测试数据集
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        testset = datasets.CIFAR100(root='./fuck', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(accuracy)
        return accuracy


###########################################################################################################
###########################################################################################################
###########################################################################################################





tested_model = model(0.2)








import random

# Genetic Algorithm to maximize the 'valuate' function output for a permutation of numbers 0-1567

# Parameters
POPULATION_SIZE = 100    # Population size
GENERATIONS = 100        # Number of generations
CROSSOVER_RATE = 0.8     # Crossover rate
MUTATION_RATE = 0.2      # Mutation rate
TOURNAMENT_SIZE = 3      # Tournament size for selection

# Length of the sequence
N = 1568

# Assuming 'valuate' function is available (no need to implement it)
# The 'valuate' function takes a permutation (list of integers) as input and returns a profit value
def valuate(permutation):
    print(permutation[:10])
    # Placeholder for the actual 'valuate' function
    # In practice, this function should be provided and will compute the profit value of the permutation
    # For demonstration purposes, we'll return a random value
    return tested_model.valuate(permutation)

# Definition of an individual in the population
class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome  # The permutation sequence
        self.fitness = None           # Fitness value (to be evaluated)

    def evaluate_fitness(self):
        if self.fitness is None:
            self.fitness = valuate(self.chromosome)
        return self.fitness

# Initialize the population with random permutations
def initialize_population():
    population = []
    base_sequence = list(range(N))
    for _ in range(POPULATION_SIZE):
        chromosome = base_sequence[:]
        random.shuffle(chromosome)
        individual = Individual(chromosome)
        population.append(individual)
    return population

# Tournament selection to choose the best individual among a random subset
def tournament_selection(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=lambda ind: ind.fitness, reverse=True)
    return tournament[0]

# Order Crossover (OX) for permutations
def order_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(N), 2))
    child_chromosome = [None] * N
    # Copy the crossover segment from parent1 to the child
    child_chromosome[start:end+1] = parent1.chromosome[start:end+1]
    # Fill the remaining positions with genes from parent2 in order
    p2_genes = parent2.chromosome[end+1:] + parent2.chromosome[:end+1]
    child_pos = (end + 1) % N
    for gene in p2_genes:
        if gene not in child_chromosome:
            child_chromosome[child_pos] = gene
            child_pos = (child_pos + 1) % N
    return Individual(child_chromosome)

# Swap Mutation for permutations
def swap_mutation(individual):
    idx1, idx2 = random.sample(range(N), 2)
    individual.chromosome[idx1], individual.chromosome[idx2] = individual.chromosome[idx2], individual.chromosome[idx1]

# The main Genetic Algorithm function
def genetic_algorithm():
    # Initialize population
    population = initialize_population()
    best_individual = None

    for generation in range(GENERATIONS):
        # Evaluate fitness for all individuals
        for individual in population:
            individual.evaluate_fitness()

        # Keep track of the best individual
        current_best = max(population, key=lambda ind: ind.fitness)
        if best_individual is None or current_best.fitness > best_individual.fitness:
            best_individual = Individual(current_best.chromosome[:])
            best_individual.fitness = current_best.fitness

        # Create a new generation
        new_population = [best_individual]  # Elitism: carry over the best individual

        while len(new_population) < POPULATION_SIZE:
            # Selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            # Crossover
            if random.random() < CROSSOVER_RATE:
                child = order_crossover(parent1, parent2)
            else:
                child = Individual(parent1.chromosome[:])  # Clone parent

            # Mutation
            if random.random() < MUTATION_RATE:
                swap_mutation(child)

            # Evaluate fitness of the new child
            child.evaluate_fitness()

            # Add child to the new population
            new_population.append(child)

        # Replace old population with the new generation
        population = new_population

        # Optionally, print current generation's best fitness
        print(f"Generation {generation+1}: Best Fitness = {best_individual.fitness}")

    # After all generations, return the best individual found
    return best_individual

# Run the Genetic Algorithm
if __name__ == '__main__':
    best_solution = genetic_algorithm()
    print("\nBest solution found:")
    print("Fitness:", best_solution.fitness)
    # Printing the entire chromosome may not be practical due to its length
    # Uncomment the following line if you wish to see the complete permutation
    # print("Permutation:", best_solution.chromosome)
