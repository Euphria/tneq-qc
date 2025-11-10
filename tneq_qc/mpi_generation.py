from typing import Any, Callable
import numpy as np
import random, string, os
from .evolve import EVOLVE_OPS, FITNESS_FUNCS
from .callbacks import CALLBACKS, LOG_FORMATER
from .mpi_core import REASONS, INDIVIDUAL_STATUS
from .tn_graph import TNGraph
import itertools
from dataclasses import dataclass, field


class Individual:
    """
    Individual represents a single solution in the genetic algorithm.
    It encapsulates the network structure (graph), identity information, 
    and performance metrics.
    """

    @staticmethod
    def full_connection_adj_matrix(individual):
        if isinstance(individual.presented_shape, list):
            adj_matrix = np.diag(individual.presented_shape)
        else:
            adj_matrix = np.diag([individual.presented_shape]*individual.tn_size)

        adj_matrix[np.triu_indices(individual.tn_size, 1)] = individual.tn_rank

        return adj_matrix

    @staticmethod
    def naive_random_adj_matrix_with_sparsity_limitation(individual):
        if isinstance(individual.presented_shape, list):
            adj_matrix = np.diag(individual.presented_shape)
        else:
            adj_matrix = np.diag([individual.presented_shape]*individual.tn_size)

        if individual.init_sparsity < 0:
            connection = []
            real_init_sparsity = np.random.uniform(low=-individual.init_sparsity, high=1.0)
            for _ in range(np.sum(np.arange(individual.tn_size))):
                connection.append(int(np.random.uniform()>real_init_sparsity)*individual.tn_rank)
        else:
            connection = [ int(np.random.uniform()>individual.init_sparsity)*individual.tn_rank for _ in range(np.sum(np.arange(individual.tn_size)))]
        adj_matrix[np.triu_indices(individual.tn_size, 1)] = connection
        return adj_matrix

    def __init__(self, 
                 scope: str,
                 graph_string: str = None,
                 parents: tuple = None,
                 mutation_prob: float = 0.1,
                 tn_rank: int = 2,
                 fitness_func: Callable = None,
                 **kwds):
        """
        Initialize an Individual.
        
        Args:
            scope: Unique identifier for this individual (e.g., "G001/SOC01/042")
            graph: Adjacency matrix representing the tensor network structure
            parents: Tuple of parent scopes (for tracking genealogy)
            mutation_prob: Probability of mutation during evolution
            tn_rank: Tensor rank for mutation operations
            fitness_func: Function to calculate fitness score
            **kwds: Additional configuration parameters
        """
        # ============================================================
        # 1. Identity Information
        # ============================================================
        self.scope = scope
        self.parents = parents if parents is not None else ()
        
        # ============================================================
        # 2. Graph Structure (Network)
        # ============================================================
        self.graph = TNGraph(graph_string)
        self.dim = self.graph.n_qubits
        
        # ============================================================
        # 3. Score-related Information (Initialized)
        # ============================================================
        # Training results
        self.report_loss = []  # List of reported losses from multiple evaluations
        self.report_loss_iter = []  # Iterations reached for each evaluation
        self.report_loss_reason = []  # Termination reasons (timeout, converged, etc.)
        
        # Estimated scores
        self.estimate_score = None  # Estimated score before evaluation
        
        # Sparsity metrics
        self.sparsity = self._calculate_sparsity()
        self.sparsity_connection = self._calculate_sparsity_connection()
        
        # Status tracking
        self.status = INDIVIDUAL_STATUS()
        
        # Fitness score (calculated after evaluation)
        self.fitness_score = None
        
        # ============================================================
        # 4. Configuration
        # ============================================================
        self.mutation_prob = mutation_prob
        self.tn_rank = tn_rank
        self.fitness_func = fitness_func if fitness_func else FITNESS_FUNCS.defualt
        self.discard_hard_timeout_result = kwds.get('discard_hard_timeout_result', False)

    def _calculate_sparsity(self) -> float:
        """
        Calculate the sparsity of the tensor network.
        Sparsity = (actual elements) / (presented elements)
        """
        # TODO: Implement sparsity calculation
        # graph_k = np.copy(self.graph)
        # graph_k[graph_k == 0] = 1
        
        # print("graph_k", graph_k)

        # present_elements = np.prod(np.diag(graph_k))
        # actual_elements = np.sum([np.prod(graph_k[d]) for d in range(self.dim)])
        
        # return actual_elements / present_elements if present_elements > 0 else 0.0

        return 0.5

    def _calculate_sparsity_connection(self) -> int:
        """
        Calculate the number of connections in the upper triangle.
        """
        # TODO: Implement connection count calculation
        # return np.sum(self.graph[np.triu_indices(self.graph.shape[0], 1)] > 0)
        return 0.5

    # ============================================================
    # Core Functions
    # ============================================================
    
    def calculate_fitness(self) -> float:
        """
        Calculate fitness score based on sparsity and best loss.
        
        Returns:
            fitness_score: Combined score (lower is better)
        """
        if not self.report_loss:
            # No evaluation results yet, return worst score
            self.fitness_score = float('inf')
        else:
            best_loss = np.min(self.report_loss)
            self.fitness_score = self.fitness_func(self.sparsity, best_loss)
        
        return self.fitness_score

    def mutate(self) -> 'Individual':
        """
        Perform mutation on the graph structure.
        Randomly flips connections in the upper triangle of the adjacency matrix.
        
        Returns:
            self: For method chaining
        """

        op = random.choice(range(3))

        print(f'mutate from graph \n{self.graph.to_string()}')

        qubit_idx = random.choice(range(self.dim))

        # TODO: change 100 -> max_trys
        success = False
        for i in range(100):
            if op == 0:
                # 选择一个link，把link数-1
                num_link = len(self.graph.graph[qubit_idx])
                link_idx = random.choice(range(num_link))

                tensor_name = self.graph.graph[qubit_idx][link_idx][0]

                try:
                    self.graph.modify_bond(qubit_idx, tensor_name, random.choice([0, self.tn_rank]))
                except ValueError:
                    continue

            elif op == 1:
                # 选择一个link，把link拆开，也就是add一个tensor
                num_link = len(self.graph.graph[qubit_idx])
                link_idx = random.choice(range(num_link))

                tensor_name = self.graph.graph[qubit_idx][link_idx][0]

                try:
                    self.graph.insert_tensor_after(qubit_idx, tensor_name)
                except ValueError:
                    continue
            else:
                # 选择一个tensor，把tensor删掉
                num_tensor = len(self.graph.graph[qubit_idx])
                tensor_idx = random.choice(range(num_tensor))

                try:
                    self.graph.remove_tensor_from_qubit(qubit_idx, self.graph.graph[qubit_idx][tensor_idx][0])
                except ValueError:
                    continue

            success = True
            print(f"successfully mutated with attempt {i}")

            break

        return self

    def crossover(self, other: 'Individual') -> tuple['Individual', 'Individual']:
        """
        Perform crossover with another individual.
        
        Args:
            other: Another Individual to crossover with
            
        Returns:
            (offspring1, offspring2): Two new individuals
            
        Note:
            This is a placeholder. Actual implementation should be added based on
            specific crossover strategy (e.g., single-point, multi-point, uniform).
        """
        # TODO: Implement crossover logic
        # Possible strategies:
        # 1. Exchange random rows between graphs
        # 2. Blend graphs with weighted average
        # 3. Multi-point crossover on upper triangle
        raise NotImplementedError("Crossover not yet implemented")

    # ============================================================
    # Training Interface Functions
    # ============================================================
    
    def get_training_info(self) -> dict:
        """
        Get information needed for training/evaluation.
        
        Returns:
            dict: Contains graph, scope, and other relevant information
        """
        return {
            'adj_matrix': self.graph,  # TNGraph object
            'graph_string': self.graph.to_string(),  # String representation for serialization
            'scope': self.scope,
            'parents': self.parents,
            'sparsity': self.sparsity,
            'dim': self.dim
        }

    def set_training_result(self, 
                           loss: float, 
                           iterations: int, 
                           reason: int = REASONS.REACH_MAX_ITER) -> bool:
        """
        Set the training result for this individual.
        
        Args:
            loss: The loss value from training
            iterations: Number of iterations completed
            reason: Termination reason (from REASONS class)
            
        Returns:
            bool: True if result was accepted, False if discarded
        """
        # Discard hard timeout results if configured
        if self.discard_hard_timeout_result and reason == REASONS.HARD_TIMEOUT:
            return False
        
        self.report_loss.append(loss)
        self.report_loss_iter.append(iterations)
        self.report_loss_reason.append(reason)
        
        # Recalculate fitness score
        self.calculate_fitness()
        
        return True

    # ============================================================
    # Utility Functions
    # ============================================================
    
    def __str__(self) -> str:
        """String representation of the individual."""

        if self.status.finished:
            opt_str = f'{self.scope}: '
            opt_str += f'sparsity={self.sparsity:.3f}, '
            opt_str += f'fitness={self.fitness_score:.5f}, '
            opt_str += f'losses={[float(f"{l:.4f}") for l in self.report_loss]}, '
            opt_str += f'parents={self.parents}\n'
            opt_str += str(self.graph) + '\n'
        else:
            opt_str = f'{self.scope}: Status={self.status}'
        
        return opt_str

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Individual(scope={self.scope}, fitness={self.fitness_score}, " \
               f"sparsity={self.sparsity:.3f}, evaluated={len(self.report_loss)}, " \
               f"losses={[float(f'{l:.4f}') for l in self.report_loss]}), graph=\n{self.graph.to_string()}\n"

    def __call__(self, action=None, *args: Any, **kwds: Any) -> Any:
        ## call an individual act as follows
        ## 1. depoly: report its adj_matrix to generation, 
        ##            generation will then forward it to overlord,
        ##            individual then tracks the rank it passed to.
        ## 2. collect: overlord report the repeat_loss to generation, 
        ##             generation forward this loss to individual,
        ##             individual process if the loss (discard of keep) then append it to repeat loss.
        ## 3. assign: overlord report problem of this individual, 
        ##            therefore generation provide a fake result for it

        if action == 'deploy':
            # return dict(adj_matrix=self.adj_matrix, scope=self.scope)
            return dict(adj_matrix=None, scope=self.scope)
        
        elif action == 'collect':
            reported_result = kwds.get('reported_result', None)
            if reported_result:
                if self.discard_hard_timeout_result and reported_result['reason'] == REASONS.HARD_TIMEOUT:
                    pass
                else:
                    self.repeat_loss.append(reported_result['loss'])
                    self.repeat_loss_iter.append(reported_result['current_iter'])
                    self.repeat_loss_reason.append(reported_result['reason'])
                return True
            else:
                return False

        elif action == 'assign':
            self.repeat_loss.append(kwds.get('loss', 1e9))
            self.repeat_loss_iter.append(-1)
            self.repeat_loss_reason.append(REASONS.FAKE_RESULT)
            return True
        else:
            return

    # ============================================================
    # Static Factory Methods (for backward compatibility)
    # ============================================================
    
    @staticmethod
    def create_full_connection(scope: str, 
                              tn_size: int = 4, 
                              tn_rank: int = 2, 
                              presented_shape: int = 2,
                              **kwds) -> 'Individual':
        """Create an individual with fully connected graph."""
        # Build a fully connected TNGraph string
        # For tn_size qubits, each qubit has all tensors A, B, C, ... connected
        # Format: -2-A--r--B--r--C--r--...-2-
        import string
        
        tensor_names = list(string.ascii_uppercase[:tn_size])
        lines = []
        
        for i in range(tn_size):
            # Each line has all tensors, fully connected
            parts = [f"-{presented_shape if isinstance(presented_shape, int) else presented_shape[i]}-"]
            for j, name in enumerate(tensor_names):
                parts.append(name)
                if j < len(tensor_names) - 1:
                    parts.append(f"--{tn_rank}--")
            parts.append(f"-{presented_shape if isinstance(presented_shape, int) else presented_shape[i]}-")
            lines.append(''.join(parts))
        
        graph_string = '\n'.join(lines)
        
        return Individual(scope=scope, graph_string=graph_string, tn_rank=tn_rank, **kwds)

    @staticmethod
    def create_random(scope: str,
                     tn_size: int = 4,
                     tn_rank: int = 2,
                     presented_shape: int = 2,
                     init_sparsity: float = 0.5,
                     **kwds) -> 'Individual':
        """Create an individual with random sparse connections."""
        import string
        import random
        
        # Determine sparsity
        if init_sparsity < 0:
            real_init_sparsity = np.random.uniform(low=-init_sparsity, high=1.0)
        else:
            real_init_sparsity = init_sparsity
        
        tensor_names = list(string.ascii_uppercase[:tn_size])
        lines = []
        
        for i in range(tn_size):
            # Build line with random connections
            parts = [f"-{presented_shape if isinstance(presented_shape, int) else presented_shape[i]}-"]
            
            for j, name in enumerate(tensor_names):
                parts.append(name)
                if j < len(tensor_names) - 1:
                    # Random connection: either tn_rank or 0 based on sparsity
                    bond_value = 0 if np.random.uniform() < real_init_sparsity else tn_rank
                    if bond_value > 0:
                        parts.append(f"--{bond_value}--")
                    else:
                        parts.append("-----")
            
            parts.append(f"-{presented_shape if isinstance(presented_shape, int) else presented_shape[i]}-")
            lines.append(''.join(parts))
        
        graph_string = '\n'.join(lines)
        
        return Individual(scope=scope, graph_string=graph_string, tn_rank=tn_rank, **kwds)


class Generation:

    @dataclass
    class Society:
        name: None
        individuals: list[Any] = field(default_factory=list)
        score_original: list[Any] = field(default_factory=list)
        score_total: list[Any] = field(default_factory=list)
        indv_ranking: list[Any] = field(default_factory=list)
        finished: bool = False
        fitness_func: Callable = None

        def __iter__(self):
            for i in self.individuals:
                yield i.scope, i

        def __len__(self):
            return len(self.individuals)
        
        def __str__(self):
            opt_str = f'===== SOCIETY {self.name} =====\n'
            if self.finished:
                # 已完成评估：显示所有个体的详细信息
                best_idx = self.indv_ranking[0] if self.indv_ranking else None
                for idx, indv in enumerate(self.individuals):
                    if idx == best_idx:  ## 最佳个体用红色标记
                        opt_str += LOG_FORMATER.RED_F.format(content=str(indv))
                    else:
                        opt_str += LOG_FORMATER.BLUE_F.format(content=str(indv))
            else:
                # 未完成：显示个体状态统计
                n_finished = sum(int(i.status.finished) for i in self.individuals)
                n_pending = len(self.individuals) - n_finished
                
                opt_str += f'Total individuals: {len(self.individuals)}\n'
                opt_str += f'Finished: {n_finished}, Pending: {n_pending}\n'
                
                # 显示状态详情
                opt_str += f'Individual status: '
                for i in self.individuals:
                    status_str = '✓' if i.status.finished else f'({len(i.report_loss)}/{len(self.individuals)})'
                    opt_str += f'[{i.scope[-7:]}: {status_str}] '
                opt_str += '\n'
            return opt_str

    def init_societies_individuals(self, pG=None):
        """
        Initialize societies and their individuals.
        
        Args:
            pG: Parent Generation (if creating from previous generation)
        """
        if pG:
            # Inherit from parent generation
            for k, v in pG.societies.items():
                society = self.Society(
                    name=k,
                    individuals=[],
                    fitness_func=v.fitness_func if hasattr(v, 'fitness_func') else None
                )
                
                # Create new individuals from parent generation
                # Use graph string from parent's TNGraph
                society.individuals = [
                    Individual(
                        scope='{}/{}/{:03d}'.format(self.name, k, idx),
                        graph_string=indv.graph.to_string(),  # Convert TNGraph to string
                        parents=(indv.scope,) if indv.parents == () else indv.parents + (indv.scope,),
                        # mutation_prob=self.kwds.get('mutation_prob', 0.1),
                        # tn_rank=self.kwds.get('tn_rank', 2),
                        fitness_func=society.fitness_func,
                        **self.kwds
                    )
                    for idx, indv in enumerate(v.individuals)
                ]
                
                self.societies[k] = society
                self.indv_to_distribute += [indv for indv in society.individuals]

        else:
            # Create from scratch (first generation)
            for param_dict in self.society_params_list:
                n_individuals = param_dict.get('n_individuals_span', 20)
                fitness_func = param_dict.get('fitness_func', FITNESS_FUNCS.defualt)
                
                # Generate random society name
                society_name = ''.join(random.choice(string.ascii_uppercase + string.digits) 
                                      for _ in range(5))
                
                society = self.Society(
                    name=society_name,
                    individuals=[],
                    fitness_func=fitness_func
                )
                
                # Create random individuals using factory method
                graph_string_template = param_dict.get('graph_string_template', None)
                
                for i in range(n_individuals):
                    scope = '{}/{}/{:03d}'.format(self.name, society_name, i)
                    
                    if graph_string_template:
                        # Use provided template
                        indv = Individual(
                            scope=scope,
                            graph_string=graph_string_template,
                            # mutation_prob=self.kwds.get('mutation_prob', 0.1),
                            # tn_rank=self.kwds.get('tn_rank', 2),
                            fitness_func=fitness_func,
                            **self.kwds
                        )
                    else:
                        # TODO: not implemented
                        # Use factory method for random creation
                        indv = Individual.create_random(
                            scope=scope,
                            # tn_size=self.kwds.get('tn_size', 4),
                            # tn_rank=self.kwds.get('tn_rank', 2),
                            # presented_shape=self.kwds.get('presented_shape', 2),
                            # init_sparsity=self.kwds.get('init_sparsity', 0.5),
                            # mutation_prob=self.kwds.get('mutation_prob', 0.1),
                            fitness_func=fitness_func,
                            **self.kwds
                        )
                    
                    society.individuals.append(indv)
                
                self.societies[society_name] = society
                self.indv_to_distribute += [indv for indv in society.individuals]

    def __init__(self, pG=None, name=None, **kwds):
        ## basic propoerties
        self.name = name

        ## parse the kwds
        self.kwds = kwds
        self.logger = kwds.get('logger', None)
        self.generation_property = kwds.get('generation_property', {})
        self.evaluate_repeat = self.generation_property.get('evaluate_repeat', 2)
        self.still_allow_repeat_after_hard_timeout = self.generation_property.get('still_allow_repeat_after_hard_timeout', True)

        ## Queues for individuals
        self.indv_to_distribute = []
        self.indv_to_collect = []

        ## parse and init societies
        self.society_property = self.generation_property.get('society_property', {})
        self.n_societies = self.generation_property.get('n_societies', 1)
        self.society_params_list = self.society_property.get('society', [dict(n_individuals_span=20, n_individuals_survive=10, fitness_func=FITNESS_FUNCS.defualt)])
        if len(self.society_params_list) == 1 and self.n_societies > 1:
            self.society_params_list = self.society_params_list * self.n_societies
        elif len(self.society_params_list) != self.n_societies:
            raise ValueError('Cannot parse society_params due to number balance between n_societies and society_params.')

        self.societies = {}
        
        self.init_societies_individuals(pG=pG)
        
        ## prepare evolve ops
        self.evolution_property = kwds.get('evolution_property', dict(elimiation_threshold=0.8))


    def evolve(self):
        """
        Execute evolution: Select top-k individuals, copy them n_copy times, and mutate.
        
        Strategy:
        1. Sort all individuals by fitness (lower is better)
        2. Select top k individuals
        3. Create n_copy copies of each selected individual
        4. Apply mutation to all copied individuals
        """
        # Get evolution parameters
        top_k = self.evolution_property.get('top_k', 5)
        n_copy = self.evolution_property.get('n_copy', 4)
        mutation_prob = self.evolution_property.get('mutation_prob', 0.2)
        
        for society_name, society in self.societies.items():
            # Step 1: Sort individuals by fitness (lower is better)
            sorted_individuals = sorted(
                society.individuals,
                key=lambda x: x.fitness_score if x.fitness_score is not None else float('inf')
            )
            
            # Step 2: Select top k individuals
            k = min(top_k, len(sorted_individuals))
            top_individuals = sorted_individuals[:k]
            
            if self.logger:
                self.logger.info(f'Society {society_name}: Selected top {k} individuals')
                for i, indv in enumerate(top_individuals):
                    self.logger.info(f'  {i+1}. {indv.scope} - fitness={indv.fitness_score:.5f}')
            
            # Step 3: Create copies of top individuals
            new_individuals = []
            offspring_counter = 0
            
            for parent in top_individuals:
                # Create n_copy copies of this individual
                for copy_idx in range(n_copy):
                    offspring_scope = '{}/{}/{:03d}'.format(
                        self.name, 
                        society_name, 
                        len(sorted_individuals) + offspring_counter
                    )
                    offspring_counter += 1
                    
                    # Create offspring by copying parent's graph
                    offspring = Individual(
                        scope=offspring_scope,
                        graph_string=parent.graph.to_string(),
                        parents=(parent.scope,),
                        # mutation_prob=mutation_prob,
                        # tn_rank=self.kwds.get('tn_rank', 2),
                        fitness_func=society.fitness_func,
                        **self.kwds
                    )
                    
                    # Step 4: Apply mutation to offspring
                    offspring.mutate()
                    
                    new_individuals.append(offspring)
            
            # Update society with new generation
            society.individuals = new_individuals
            
            if self.logger:
                self.logger.info(
                    f'Society {society_name}: Created {len(new_individuals)} new individuals '
                    f'({k} parents × {n_copy} copies)'
                )

    def evaluate(self):
        """
        Evaluate all individuals in all societies and rank them.
        """
        for society_name, society in self.societies.items():
            # Calculate fitness for all individuals
            scores = []
            for indv in society.individuals:
                if indv.report_loss:
                    # Individual has been evaluated
                    indv.calculate_fitness()
                    scores.append(indv.fitness_score)
                else:
                    # Not yet evaluated
                    scores.append(float('inf'))
            
            # Rank individuals (lower score is better)
            society.score_total = scores
            society.indv_ranking = list(np.argsort(scores))
            
            if self.logger:
                best_idx = society.indv_ranking[0] if society.indv_ranking else None
                if best_idx is not None and scores[best_idx] != float('inf'):
                    best_indv = society.individuals[best_idx]
                    self.logger.info(
                        f'Society {society_name}: Best individual {best_indv.scope} '
                        f'with fitness {best_indv.fitness_score:.5f}'
                    )

    def distribute_indv(self, agent):
        """
        Distribute an individual to an agent for evaluation.
        
        Args:
            agent: MPI agent to receive the individual
        """
        if not self.indv_to_distribute:
            if self.logger:
                self.logger.debug('No individuals to distribute')
            return False
        
        indv = self.indv_to_distribute.pop(0)
        
        # Check sparsity threshold
        sparsity_threshold = self.generation_property.get('sparsity_threshold', 10.0)
        if np.log10(indv.sparsity) < sparsity_threshold:
            # Sparsity is acceptable, send to agent
            training_info = indv.get_training_info()
            agent.receive(training_info)
            
            self.indv_to_collect.append(indv)
            
            if self.logger:
                self.logger.info(
                    f'Assigned individual {indv.scope} to agent {agent.rank}, '
                    f'sparsity={indv.sparsity:.3f}'
                )
            return True
        else:
            # Sparsity too high, assign fake result
            indv.set_training_result(
                loss=1e9,
                iterations=0,
                reason=REASONS.FAKE_RESULT
            )
            
            if self.logger:
                self.logger.warning(
                    f'Individual {indv.scope} rejected due to high sparsity '
                    f'(log10={np.log10(indv.sparsity):.2f}, connections={indv.sparsity_connection})'
                )
            
            # Try next individual
            return self.distribute_indv(agent)

    def collect_indv(self, result_data: dict):
        """
        Collect evaluation results and assign to corresponding individual.
        
        Args:
            result_data: Dictionary containing:
                - scope: Individual identifier
                - loss: Training loss value
                - iterations: Number of iterations completed
                - reason: Termination reason
        """
        scope = result_data.get('scope')
        loss = result_data.get('loss')
        iterations = result_data.get('iterations', -1)
        reason = result_data.get('reason', REASONS.REACH_MAX_ITER)
        
        # Find the individual in collection queue
        for indv in self.indv_to_collect:
            if indv.scope == scope:
                # Set training result
                accepted = indv.set_training_result(loss, iterations, reason)
                
                if accepted:
                    if self.logger:
                        self.logger.info(
                            f'Collected result for {scope}: loss={loss:.4f}, '
                            f'iterations={iterations}, reason={REASONS.__rdict__.get(reason, reason)}'
                        )
                    
                    # Check if this individual needs more evaluations
                    if len(indv.report_loss) >= self.evaluate_repeat:
                        # Sufficient evaluations, mark as finished
                        indv.status.finished = True
                        self.indv_to_collect.remove(indv)
                        
                        if self.logger:
                            self.logger.info(
                                f'Individual {scope} completed {len(indv.report_loss)} evaluations'
                            )
                    else:
                        # Need more evaluations, move back to distribute queue
                        self.indv_to_collect.remove(indv)
                        self.indv_to_distribute.append(indv)
                        
                        if self.logger:
                            self.logger.debug(
                                f'Individual {scope} needs more evaluations '
                                f'({len(indv.report_loss)}/{self.evaluate_repeat})'
                            )
                else:
                    if self.logger:
                        self.logger.warning(f'Result for {scope} was discarded')
                
                return True
        
        if self.logger:
            self.logger.warning(f'Could not find individual {scope} in collection queue')
        return False
    
    def collect_all_pending(self):
        """
        Check all individuals in collection queue for any completed evaluations.
        This is useful when results arrive asynchronously.
        """
        completed_count = 0
        for indv in list(self.indv_to_collect):  # Copy list to avoid modification during iteration
            if len(indv.report_loss) >= self.evaluate_repeat:
                indv.status.finished = True
                self.indv_to_collect.remove(indv)
                completed_count += 1
        
        if completed_count > 0 and self.logger:
            self.logger.info(f'Marked {completed_count} individuals as completed')
        

    def is_an_individual_finished(self, indv) -> bool:
        """
        Check if an individual has completed all required evaluations.
        
        Args:
            indv: Individual to check
            
        Returns:
            bool: True if finished
        """
        if indv.status.finished:
            return True
        
        # Check if sufficient evaluations completed
        if len(indv.report_loss) >= self.evaluate_repeat:
            indv.status.finished = True
            return True
        
        # Check hard timeout policy
        if not self.still_allow_repeat_after_hard_timeout:
            for reason in indv.report_loss_reason:
                if reason == REASONS.HARD_TIMEOUT:
                    indv.status.finished = True
                    return True
        
        return False

    def is_a_society_finished(self, society) -> bool:
        """
        Check if all individuals in a society have finished evaluation.
        
        Args:
            society: Society to check
            
        Returns:
            bool: True if all individuals finished
        """
        for indv in society.individuals:
            if not self.is_an_individual_finished(indv):
                return False
        
        society.finished = True
        return True

    def is_finished(self) -> bool:
        """
        Check if the entire generation has completed evaluation and evolution.
        
        Returns:
            bool: True if generation is finished
        """
        if self.logger:
            self.logger.debug(
                f'Checking if generation {self.name} is finished: '
                f'to_distribute={len(self.indv_to_distribute)}, '
                f'to_collect={len(self.indv_to_collect)}'
            )

        # Check if all queues are empty
        if len(self.indv_to_distribute) == 0 and len(self.indv_to_collect) == 0:
            # All individuals have been evaluated
            
            if self.logger:
                self.logger.info(f'Generation {self.name}: All individuals evaluated')
            
            # Perform evaluation and ranking
            self.evaluate()
            
            # Perform evolution operations
            self.evolve()
            
            # Mark all societies as finished
            for society in self.societies.values():
                society.finished = True
            
            if self.logger:
                self.logger.info(f'Generation {self.name} finished evaluation and evolution')
            
            return True
        
        return False
    
    def get_best_individual(self):
        """
        Get the best individual across all societies.
        
        Returns:
            Individual: Best performing individual (lowest fitness score)
        """
        best_indv = None
        best_score = float('inf')
        
        self.logger.debug(f'Getting best individual in generation {self.name} from societies {self.societies}.')
        self.logger.debug(f"Generation {self.name} societies' details: {self.societies.values()}")

        for society in self.societies.values():
            self.logger.debug(f'Checking society {society.name} for best individual, indv_ranking={society.indv_ranking} and individuals count={len(society.individuals)}')
            if society.indv_ranking and society.individuals:
                best_idx = society.indv_ranking[0]
                indv = society.individuals[best_idx]
                
                self.logger.debug(f'Checking individual {indv.scope} with fitness {indv.fitness_score}')

                if indv.fitness_score and indv.fitness_score < best_score:
                    best_score = indv.fitness_score
                    best_indv = indv
        
        return best_indv
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the generation.
        
        Returns:
            dict: Statistics including best, worst, mean fitness scores
        """
        all_scores = []
        for society in self.societies.values():
            for indv in society.individuals:
                if indv.fitness_score and indv.fitness_score != float('inf'):
                    all_scores.append(indv.fitness_score)
        
        if not all_scores:
            return {
                'n_individuals': sum(len(s.individuals) for s in self.societies.values()),
                'n_evaluated': 0,
                'best_score': None,
                'worst_score': None,
                'mean_score': None,
                'std_score': None
            }
        
        return {
            'n_individuals': sum(len(s.individuals) for s in self.societies.values()),
            'n_evaluated': len(all_scores),
            'best_score': np.min(all_scores),
            'worst_score': np.max(all_scores),
            'mean_score': np.mean(all_scores),
            'std_score': np.std(all_scores)
        }

    def __str__(self) -> str:
        """String representation of the generation."""
        lines = [f'===== GENERATION {self.name} =====']
        lines.append(f'Societies: {len(self.societies)}')
        lines.append(f'To distribute: {len(self.indv_to_distribute)}')
        lines.append(f'To collect: {len(self.indv_to_collect)}')
        
        stats = self.get_statistics()
        if stats['n_evaluated'] > 0:
            lines.append(f"Evaluated: {stats['n_evaluated']}/{stats['n_individuals']}")
            lines.append(f"Best score: {stats['best_score']:.5f}")
            lines.append(f"Mean score: {stats['mean_score']:.5f} ± {stats['std_score']:.5f}")
        
        for society_name, society in self.societies.items():
            lines.append(f'\n{society}')
        
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Generation(name={self.name}, societies={len(self.societies)}, " \
               f"individuals={sum(len(s.individuals) for s in self.societies.values())})"

    def __report_generation__(self):
        self.logger.info('Current length of indv_to_distribute is {}.'.format(len(self.current_generation.indv_to_distribute)))
        self.logger.info('Current length of indv_to_collect is {}.'.format(len(self.current_generation.indv_to_collect)))
        self.logger.info([(indv.scope, indv.sge_job_id) for indv in self.current_generation.indv_to_collect])


    def __report_agents__(self):
        self.logger.info('Current number of known agents is {}.'.format(len(self.known_agents)))
        self.logger.info(list(self.known_agents.keys()))



    def __call__(self, action: str = None, *args, **kwds):
        """
        Callable interface for generation operations.
        
        Args:
            action: Action to perform ('evaluate', 'evolve', 'status', etc.)
        """
        if action == 'evaluate':
            self.evaluate()
            return True
        
        elif action == 'evolve':
            self.evolve()
            return True
        
        elif action == 'status':
            return self.get_statistics()
        
        elif action == 'best':
            return self.get_best_individual()
        
        else:
            # Default: check if finished and auto-evolve
            if self.is_finished():
                return {'finished': True, 'stats': self.get_statistics()}
            else:
                return {'finished': False, 
                       'to_distribute': len(self.indv_to_distribute),
                       'to_collect': len(self.indv_to_collect)}