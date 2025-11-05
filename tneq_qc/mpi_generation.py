from typing import Any, Callable
import numpy as np
import random, string, os
from .evolve import EVOLVE_OPS, FITNESS_FUNCS
from .callbacks import CALLBACKS, LOG_FORMATER
from .mpi_core import REASONS, INDIVIDUAL_STATUS
import itertools
from dataclasses import dataclass, field


class Individual:

    ## Creation of adj matrix based on function now is method of individual.
    ## functions now is called by Generation by staticmethod@individual

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

    def __init__(self, adj_matrix=None, scope=None, **kwds):
        # Parse kwds first to get parameters needed for matrix generation
        self.scope = scope
        self.parents = kwds.get('parents', None)
        self.individual_property = kwds.get('individual_property', {})
        self.discard_hard_timeout_result = self.individual_property.get('discard_hard_timeout_result', False)
        self.random_initilization_property = self.individual_property.get('random_initilization_property', {})

        ## for random initilization
        self.tn_size = self.random_initilization_property.get('tn_size', 4)
        self.tn_rank = self.random_initilization_property.get('tn_rank', 2) 
        self.presented_shape = self.random_initilization_property.get('presented_shape', 2)
        self.init_sparsity = self.random_initilization_property.get('init_sparsity', -0.00001)
        
        ## Create adj_matrix
        if adj_matrix is None:
            adj_func = kwds.get('adj_func', Individual.naive_random_adj_matrix_with_sparsity_limitation)
            self.adj_matrix = adj_func(self)
        elif callable(adj_matrix):
            self.adj_matrix = adj_matrix(self)
        else:
            self.adj_matrix = adj_matrix

        # Make matrix symmetric
        self.dim = self.adj_matrix.shape[0]
        self.adj_matrix[np.tril_indices(self.dim, -1)] = self.adj_matrix.transpose()[np.tril_indices(self.dim, -1)]

        ## Initialize tracking variables
        self.repeat_loss = []
        self.repeat_loss_iter = []
        self.repeat_loss_reason = []
        self.total_score = None
        self.status = INDIVIDUAL_STATUS() # the running status

        ## Calculate sparsity
        ## adj_matrix_k put all the 0 edge to 1, this is only used for calulation of sparsity
        ## sparsity is the ratio of actual # elements to its presented # elements
        ## sparsity_connection is the # connections (compared to full connection)
        adj_matrix_k = np.copy(self.adj_matrix)
        adj_matrix_k[adj_matrix_k==0] = 1
        self.present_elements = np.prod(np.diag(adj_matrix_k))
        self.actual_elements = np.sum([ np.prod(adj_matrix_k[d]) for d in range(self.dim) ])
        self.sparsity = self.actual_elements / self.present_elements
        self.sparsity_connection = np.sum(self.adj_matrix[np.triu_indices(self.adj_matrix.shape[0], 1)] > 0)

    def __str__(self) -> str:
        opt_str = ''
        if self.status.finished:
            opt_str += f'{self.scope}: ' \
                       f'sparsity = {self.sparsity:.3f}, ' \
                       f'repeat_loss = {[ float("{:0.4f}".format(l)) for l in self.repeat_loss ]}, ' \
                       f'total_score = {self.total_score:.5f}, ' \
                       f'parents = {self.parents}.\n' 
            opt_str += str(self.adj_matrix)
            opt_str += '\n'
        else:
            opt_str = str(self.status)
        return opt_str

    def cal_total_score(self, fitness_func):
        self.total_score = fitness_func(self.sparsity, np.min(self.repeat_loss))
        return

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
            return dict(adj_matrix=self.adj_matrix, scope=self.scope)
        
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
                for idx, indv in enumerate(self.individuals):
                    if idx == self.indv_ranking[0]: ## the best individual
                        opt_str += LOG_FORMATER.RED_F.format(content=str(indv))
                    else:
                        opt_str += LOG_FORMATER.BLUE_F.format(content=str(indv))
            else:
                opt_str += f'Current length of indv_to_distribute is {sum(int(i.status.fininshed) for i in self.individuals)}.\n'
                opt_str += f'Current length of indv_to_collect is {sum(len(i.status.assigned) for i in self.individuals)}.\n'
                opt_str += f'Current assigned job list: '
                for i in self.individuals:
                    opt_str += f'[{i.scope} => {i.status.assigned}] '
                opt_str += '\n'
            return opt_str

    def init_societies_individuals(self, pG=None):
        if pG:
            for k, v in pG.societies.items():
                society = self.Society(
                    name=k,
                    individuals=[],
                    fitness_func=v.fitness_func if hasattr(v, 'fitness_func') else None
                )
                society.individuals = \
                        [ Individual(adj_matrix=indv.adj_matrix, parents=indv.parents,
                          scope='{}/{}/{:03d}'.format(self.name, k, idx), **self.kwds) \
                        for idx, indv in enumerate(v.individuals) ]
                self.societies[k] = society
                self.indv_to_distribute += [indv for indv in society.individuals]

        else:
            for param_dict in self.society_params_list:
                n_individuals = param_dict.get('n_individuals_span', 20)
                fitness_func = param_dict.get('fitness_func', None)
                
                society_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
                
                society = self.Society(
                    name=society_name,
                    individuals=[],
                    fitness_func=fitness_func
                )
                society.individuals = [ \
                        Individual(scope='{}/{}/{:03d}'.format(self.name, society_name, i), 
                        adj_func=Individual.naive_random_adj_matrix_with_sparsity_limitation, **self.kwds) \
                        for i in range(n_individuals) ]
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
        # ELIMINATION
        if 'elimiation_threshold' in self.kwds:
            for idx, (k, v) in enumerate(self.societies.items()):
                EVOLVE_OPS.elimination(v, self.kwds['elimiation_threshold'][idx])

        # IMMIRATION
        if 'immigration_prob' in self.kwds:
            if np.random.uniform()<self.kwds['immigration_prob']:
                EVOLVE_OPS.immigration(random.sample([v['indv'] for k, v in self.societies.items()], k=2), self.kwds['immigration_number'])

        # CROSSOVER
        if 'crossover_alpha' in self.kwds:
            for idx, (k, v) in enumerate(self.societies.items()):
                EVOLVE_OPS.crossover(v, self.kwds['population'][idx], self.kwds['crossover_alpha'])

        # MUTATION
        if 'mutation_prob' in self.kwds:
            for k, v in self.societies.items():
                for indv in v['indv']:
                    EVOLVE_OPS.mutation(indv, self.kwds['mutation_prob'])

    def evaluate(self):

        def score2rank(society, idx):
            score = society['score']
            sparsity_score = [ s for s, _ in score ]
            loss_score = [ l for _, l in score ]

            if 'fitness_func' in self.kwds.keys():
                if isinstance(self.kwds['fitness_func'], list):
                    fitness_func = self.kwds['fitness_func'][idx]
                else:
                    fitness_func = self.kwds['fitness_func']
            else:
                fitness_func = lambda s, l: s+100*l
            total_score = [ fitness_func(s, l) for s, l in zip(sparsity_score, loss_score) ]

            society['rank'] = np.argsort(total_score)
            society['total'] = total_score

        # RANKING
        for idx, (k, v) in enumerate(self.societies.items()):
            # v['score'] = [ (indv.sparsity ,np.min(indv.repeat_loss)) for indv in v['indv'] ]
            score2rank(v, idx)

    def distribute_indv(self, agent):
        if self.indv_to_distribute:
            indv = self.indv_to_distribute.pop(0)
            if np.log10(indv.sparsity)<1.0:
                agent.receive(indv)
                self.indv_to_collect.append(indv)
                self.logger.info('Assigned individual {} to agent {}.'.format(indv.scope, agent.sge_job_id))
            else:
                indv.collect(fake_loss=True)
                self.logger.info('Individual {} is killed due to its sparsity = {} / {}.'.format(indv.scope, np.log10(indv.sparsity), indv.sparsity_connection))

    def collect_indv(self):
        ## TODO
        # for indv in self.indv_to_collect:
        #     if indv.collect():
        #         self.logger.info('Collected individual result {}.'.format(indv.scope))
        #         self.indv_to_collect.remove(indv)
        pass

    def is_an_individual_finished(self, indv):
        ## To determint whether an individual being finished is according to its generation
        ## if generation find indv.status.repeat equals configuartion.repeat,
        ## or this individual has hard timeout and still_allow_repeat_after_hard_timeout is False
        ## then the individual is consider to be finished.
        ## The generation will change the indv.status.finished to True. 
        if indv.status.finished:
            return True
        else:
            pass
            # if indv.status.repeat >= 

            # if 

            return indv.status.finished
        pass

    def is_a_society_finished(self, society):
        pass

    def is_finished(self):

        # TODO: if self.previous_generation is not None:
        # self.current_generation(**self.kwds)
        # CALLBACKS.GENERATION(generation=self.current_generation)

        self.logger.debug(f'Checking if generation {self.name} is finished {len(self.indv_to_distribute)} {len(self.indv_to_collect)}')

        if len(self.indv_to_distribute) == 0 and len(self.indv_to_collect) == 0:
            # All individuals have been processed
            # Now evaluate and evolve
            self.evaluate()
            self.evolve()
            
            if self.logger:
                self.logger.info(f'Generation {self.name} finished evaluation and evolution.')
            
            return True
        else:
            return False

    def __str__(self) -> str:
        pass

    def __report_generation__(self):
        self.logger.info('Current length of indv_to_distribute is {}.'.format(len(self.current_generation.indv_to_distribute)))
        self.logger.info('Current length of indv_to_collect is {}.'.format(len(self.current_generation.indv_to_collect)))
        self.logger.info([(indv.scope, indv.sge_job_id) for indv in self.current_generation.indv_to_collect])


    def __report_agents__(self):
        self.logger.info('Current number of known agents is {}.'.format(len(self.known_agents)))
        self.logger.info(list(self.known_agents.keys()))



    def __call__(self, action, *args, **kwds):

        ## When generation is called:
        ## 1. check is_finished.
        ## 2. if finished, do evaluate and evolve, then report.
        ## 3. if not finished, report the schedule table.
        ## otherwise it acts as asked.

        try:
            self.evaluate()
            self.evolve()
            return True
        except Exception as e:
            raise e