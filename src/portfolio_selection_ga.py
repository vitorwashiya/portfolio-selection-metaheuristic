import pandas as pd
import numpy as np
from scipy import stats

POP_SIZE = 20
MUT_RATE = 0.5
MUTA_PERC = 0.5
MAX_ITER = 6000
MAX_ITER_WO_IMPRV = 100
IMPRV_TOL = 1e-8
RISK_AVER = 0.5
FIT_FUN = "mean_variance"


class PortfolioSelectionGA:
    '''
    Portfolio Selection using Genetic Algorithm - Minimization problem

    Parameters:
    - data: Dataframe with daily stock returns (index="Date", columns=["Stock.Close"])
    - num_ast: Number of assets to be selected (Default = None)
    - exp_ret_df: Expected Returns Dataframe (Default = None)
    - cov_mat: Covariance Matrix Dataframe (Default = None)
    - popl_size: Population Size (Default = 100)
    - muta_rate: Mutation Rate (Default = 0.5)
    - muta_perc: Mutation Percentage (Default = 0.5)
    - max_iter: Maximum Number of Iterations (Default = 1000)
    - max_iter_wo_imprv: Maximum Number of Iterations without Improvement (Default = 100)
    - imprv_tol: Improvement Tolerance (Default = 1e-8)
    - risk_aver: Risk Aversion Coefficient (Default = 0.5)
    - norm_param: Normalization Parameters with the following keys: ret_min, ret_max, var_min, var_max (Default = None)
    - fit_fun: Fitness Function (Default = "mean_variance")
    - verbose: Verbose mode (Default = False)
    '''

    def __init__(self,
                 data: pd.DataFrame,
                 num_ast: int = None,
                 exp_ret_df: pd.Series = None,
                 cov_mat: pd.DataFrame = None,
                 popl_size: int = POP_SIZE,
                 muta_rate: float = MUT_RATE,
                 max_iter: int = MAX_ITER,
                 max_iter_wo_imprv: int = MAX_ITER_WO_IMPRV,
                 imprv_tol: float = IMPRV_TOL,
                 risk_aver: float = RISK_AVER,
                 norm_param: dict = None,
                 fit_fun: str = FIT_FUN,
                 verbose: bool = False):
        self.num_ast = num_ast if num_ast is not None else len(data.columns)
        self.exp_ret_df = exp_ret_df if exp_ret_df is not None else data.mean()
        self.cov_mat = cov_mat if cov_mat is not None else data.cov()
        self.popl_size = popl_size
        self.muta_rate = muta_rate
        self.muta_perc = 0.5
        self.max_iter = max_iter
        self.max_iter_wo_imprv = max_iter_wo_imprv
        self.imprv_tol = imprv_tol
        self.risk_aver = risk_aver
        self.norm_param = norm_param
        self.fit_fun = fit_fun

        self.verbose = verbose
        self.iter = 0
        self.iter_wo_imprv = 0
        self.prv_best_fit = np.inf
        self.popl = None
        self.popl_fit = None
        self.initialize_population()

    def initialize_population(self) -> None:
        """
        Initializes the population for the genetic algorithm.

        This method generates a population of individuals for the genetic algorithm
        using the Dirichlet distribution. Each individual represents a portfolio
        allocation (sums up to 1) and is represented as a vector of weights for each asset.

        Returns:
            self.popl: Numpy array with the population of individuals.
        """
        #heuristics
        if self.risk_aver == 0.0:
            self.popl = np.zeros((self.popl_size, self.num_ast))
            self.popl[:, self.exp_ret_df.argmax()] = 1
        self.popl = np.random.dirichlet(np.ones(self.num_ast), self.popl_size)

    def mean_variance(self, weights) -> float:
        """
        Calculates the mean-variance fitness value of a given set of portfolio weights.

        Parameters:
        weights (array-like): The weights of the portfolio assets.

        Returns:
        float: The fitness value of the portfolio.
        """
        w_exp_ret = np.dot(weights, self.exp_ret_df)
        w_var = np.linalg.multi_dot([weights, self.cov_mat, weights])

        if self.norm_param:
            w_exp_ret = (w_exp_ret - self.norm_param["ret_min"]) / (
                self.norm_param["ret_max"] - self.norm_param["ret_min"])
            w_var = (w_var - self.norm_param["var_min"]) / (
                self.norm_param["var_max"] - self.norm_param["var_min"])

        fit = self.risk_aver * w_var + (self.risk_aver - 1) * w_exp_ret
        return fit

    def fitness(self, weights) -> float:
        """
        Calculates the fitness value of a given set of portfolio weights.

        Parameters:
        weights (array-like): The weights of the portfolio assets.

        Returns:
        float: The fitness value of the portfolio.
        """
        if self.fit_fun == "mean_variance":
            return self.mean_variance(weights)
        else:
            raise ValueError(
                "Invalid fitness function. Choose from ['mean_variance'].")

    def map_fitness(self, popl_list: np.array) -> np.array:
        """
        Maps the fitness function over a population list.

        Parameters:
        popl_list (np.array): The population list to evaluate.

        Returns:
        np.array: An array containing the fitness values for each individual in the provided population.
        """
        return np.array([self.fitness(w) for w in popl_list])

    def random_population_sample(self, population, smpl_size: int = 2) -> list:
        """
        Randomly samples individuals from the population based on their fitness percentile scores.

        Args:
            population (list): A list of individuals in the population.
            smpl_size (int, optional): The number of individuals to be drawn from the population. (Defaults to 2 - parents).

        Returns:
            list: A list of randomly sampled individuals from the population.
        """
        population_fit = self.map_fitness(population)
        perc_prob = np.array([
            1 / stats.percentileofscore(population_fit, ind)
            for ind in population_fit
        ])
        perc_prob /= perc_prob.sum()
        rnd_idx = np.random.choice(len(population),
                                   size=smpl_size,
                                   p=perc_prob,
                                   replace=False)
        return population[rnd_idx]

    def crossover(self, parents: np.array) -> np.array:
        """
        Performs crossover operation on the given parents. The crossover operation is performed by
        randomly selecting a beta value and calculating the child individual as a linear combination
        of the parent individuals.

        Parameters:
        parents (np.array): An array containing the parent individuals.

        Returns:
        np.array: The child individual resulting from the crossover operation.
        """
        beta = np.random.random()
        child = beta * parents[0] + (1 - beta) * parents[1]
        return child

    def mutate(self, child: np.array) -> list:
        """
        Mutates the given child by randomly swapping two elements in the array adding them together and setting 
        the other to zero.

        Parameters:
        child (np.array): The child to be mutated.

        Returns:
        list: A list containing the mutated child and two additional mutated children if the mutation rate is met.
        """
        children = [child.copy()]
        if np.random.random() < self.muta_rate:
            muta = 2 * [child.copy()]
            for _ in range(np.floor(MUTA_PERC * self.num_ast).astype(int)):
                pos1, pos2 = np.random.choice(self.num_ast,
                                              size=2,
                                              replace=False)
                muta[0][pos1], muta[0][pos2] = muta[0][pos1] + muta[0][pos2], 0
                muta[1][pos1], muta[1][pos2] = 0, muta[1][pos1] + muta[1][pos2]
            children.extend(muta)
        return children

    def produce_children(self) -> np.array:
        """
        Generates a new population of children by performing crossover and mutation operations on the current population.

        Returns:
            np.array: An array containing the newly generated children population.
        """
        children_popl = []
        while len(children_popl) < self.popl_size:
            parents = self.random_population_sample(population=self.popl)
            child = self.crossover(parents)
            children = self.mutate(child)
            children_popl.extend(children)
        return np.array(children_popl)

    def produce_next_gen(self):
        """
        Produces the next generation of the population by combining children and parents.

        This method selects the next generation by choosing half of the individuals from the children population and the other half from the parents population.
        The children are generated by applying genetic operators such as crossover and mutation to the current population.
        The parents are randomly selected from the current population.
        The children and parents are then combined to form the next generation of the population.

        Returns:
            self.popl: The next generation of the population.
        """
        children_popl = self.produce_children()
        num_children = int(np.floor(self.popl_size / 2))
        next_popl_child = self.random_population_sample(children_popl,
                                                        smpl_size=num_children)

        num_parents = self.popl_size - num_children
        next_popl_parent = self.random_population_sample(self.popl,
                                                         smpl_size=num_parents)
        self.popl = np.concatenate((next_popl_child, next_popl_parent))

    def stop_criteria(self) -> bool:
        """
        Check if the stopping criteria for the genetic algorithm has been met.
        The stopping criteria is met when the standard deviation of the fitness values of the population
        is below a certain threshold or the maximum number of iterations has been reached.

        Returns:
            bool: True if the stopping criteria has been met, False otherwise.
        """
        if self.iter_wo_imprv > self.max_iter_wo_imprv:
            return True
        if self.iter > self.max_iter:
            return True
        return False

    def step(self):
        """
        Performs a single step of the genetic algorithm.

        This method updates the state of the genetic algorithm by evaluating the fitness of the current population,
        determining if there has been an improvement in the best fitness value, and updating the iteration count.

        Returns:
            None
        """
        best_fit = np.min(self.popl_fit)
        if self.prv_best_fit > best_fit and abs(
                best_fit - self.prv_best_fit) > self.imprv_tol:
            self.prv_best_fit = np.min(self.popl_fit)
            self.iter_wo_imprv = 0
        else:
            self.iter_wo_imprv += 1
        self.iter += 1

    def optimize(self):
        """
        Optimizes the population using a genetic algorithm.

        This method iteratively evolves the population until a stop criteria is met.
        It produces the next generation, evaluates the fitness of the population,
        and updates the iteration count. Finally, it returns the individual with the
        lowest fitness value.

        Returns:
            numpy.ndarray: The individual with the lowest fitness value.

        """
        self.popl_fit = self.map_fitness(self.popl)
        while not self.stop_criteria():
            self.produce_next_gen()
            self.popl_fit = self.map_fitness(self.popl)
            self.step()
            if self.verbose:
                print(
                    f"Iteration: {self.iter} - Best Fitness: {np.min(self.popl_fit)}"
                )
        return self.popl[np.argmin(self.popl_fit)]
