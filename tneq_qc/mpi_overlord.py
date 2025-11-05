import os
import time
import mpi4py
import numpy as np
import functools, itertools
from .mpi_generation import Generation, Individual
from .mpi_core import TAGS, REASONS, DUMMYFUNC, AGENT_STATUS, SURVIVAL
from .callbacks import CALLBACKS
from collections import deque

class MPI_Overlord():

    ## OVERLORD SEND:
    ## 1. survival ping
    ## 2. job data

    ## OVERLORD RECEIVE
    ## 1. survival info
    ## 2. abnormal when receiving job
    ## 3. estimation info
    ## 4. job result report

    ## SINGLE JOB PIPELINE
    ## 1. overlord match an agent and an individual
    ## 2. overlord ask the individual to submit the data
    ## 3. overlord isend the data to agent
    ## 4. overlord receive result and report to the individual


    ################ UTILS ################
    def __init__(self, comm: mpi4py.MPI.COMM_WORLD, **kwds) -> None:
        self.kwds = kwds
        self.logger = kwds['logger']
        self.time = 0
        self.comm = comm
        self.agent_size = self.comm.Get_size() - 1
        self.host_status = SURVIVAL.HOST_RUNNING

        # generation
        self.max_generation = kwds['experiment']['max_generation']
        self.collection_of_generations = []

        # agents
        self.agent_report_buffer = deque()
        self.available_agents = dict(
            itertools.zip_longest(list(range(1, self.agent_size+1)), [], fillvalue=AGENT_STATUS()))

        self.logger.debug(f"Initialized MPI_Overlord with kwds: {self.kwds}")

    def tik_and_sleep(self, sec):
        self.time += sec
        time.sleep(sec)

    def call_with_interval(self, func, interval):
        if self.time % interval == 0:
            return func
        else:
            return DUMMYFUNC
        
    ################ MPI COMMUNICATION ################
    def sync_goal(self):
        """Broadcast QCTN target to all agents"""
        if not hasattr(self, 'evoluation_goal'):
            # Load from file if not already set
            goal_path = self.kwds['experiment']['evoluation_goal']
            if goal_path and os.path.exists(goal_path):
                self.evoluation_goal = np.load(goal_path)
            else:
                # Assume it's a QCTN object that will be set directly
                self.logger.warning("No evolution goal file found, will use directly set object")
        
        # Broadcast the QCTN target
        self.evoluation_goal = self.comm.bcast(self.evoluation_goal, root=0)
        self.logger.info(f"Broadcast QCTN target to all agents")

    def process_msg_surv(self, msg):
        rank_surv = msg['rank']

        self.available_agents[rank_surv].tik_time = msg['time']
        self.available_agents[rank_surv].up_time = msg['real_up_time']
        self.available_agents[rank_surv].current_iter = msg['current_iter']

        if msg['busy']:
            self.logger.info(f'Received survival report from agent rank {rank_surv}, ' \
                            f'reported tik time {msg["time"]}, real up time {msg["real_up_time"]},' \
                            f'current completion rate {msg["current_iter"]} / {msg["max_iter"]}.')
        else:
            self.logger.info(f'Received survival report from agent rank {rank_surv}, ' \
                            f'reported tik time {msg["time"]}, real up time {msg["real_up_time"]},' \
                            f'not working currently.')

        return

    def process_msg_estm(self, msg):
        rank_estm = msg['rank']

        self.available_agents[rank_estm].estimation_time = msg['required_time']

        self.logger.info(f'Received estimation report from agent rank {rank_estm}, ' \
                        f'required time is about {msg["required_time"]} with current loss {msg["loss"]}.')

        return

    def process_msg_abnm(self, msg):
        rank_abnm = msg

        self.available_agents[rank_abnm].abnormal_counter += 1

        self.logger.info(f'Received abnormal report from agent rank {rank_abnm}, ' \
                         f'current abnormal count for agent rank {rank_abnm} is ' \
                         f'{self.available_agents[rank_abnm].abnormal_counter}.')

        return

    def process_msg_rept(self, msg):

        ## Lazy process 
        ## forward the message to a deque
        ## then ask the generation to process it later

        self.agent_report_buffer.append(msg)
        rank_rept = msg['rank']

        self.available_agents[rank_rept].assigned_job = None
        self.available_agents[rank_rept].estimation_time = None
        self.available_agents[rank_rept].current_iter = None

        self.logger.info(f'Received final report from agent rank {rank_rept} with reason {REASONS.__rdict__[msg["reason"]]}, ' \
                        f'reported final iteration {msg["current_iter"]}, ' \
                        f'final loss {msg["loss"]}.')
        
        return

    def broadcast_finish(self):
        ## by default the while loop will change the status to normal shutdown
        ## therefore once the while loop is broken, should change it to abnormal
 
        if self.host_status == SURVIVAL.HOST_RUNNING:
            self.host_status = SURVIVAL.HOST_ABNORMAL_SHUTDOWN
        self.call_agent_survivability()
        return

    def call_agent_survivability(self):
        for agent in self.available_agents.keys():
            self.comm.isend(self.host_status, dest=agent, tag=TAGS.INFO_SURVIVAL)
        return

        
    def tik_and_collect_everything_from_agent(self, sec):
        ## The tik function is a while loop checking the irecv status

        self.time += sec
        start_time = time.time()
        while time.time() - start_time < sec:

            status_surv, msg_surv = self.req_surv.test()
            if status_surv:
                self.logger.debug(f"Received survival message from agent with message: {msg_surv}.")

                self.process_msg_surv(msg_surv)
                self.req_surv = self.comm.irecv(tag=TAGS.INFO_SURVIVAL)

            status_estm, msg_estm = self.req_estm.test()
            if status_estm:
                self.logger.debug(f"Received estimation message from agent with message: {msg_estm}.")

                self.process_msg_estm(msg_estm)
                self.req_estm = self.comm.irecv(tag=TAGS.INFO_TIME_ESTIMATION)

            status_abnm, msg_abnm = self.req_abnm.test()
            if status_abnm:
                self.logger.debug(f"Received abnormal message from agent with message: {msg_abnm}.")  

                self.process_msg_abnm(msg_abnm)
                self.req_abnm = self.comm.irecv(tag=TAGS.INFO_ABNORMAL)

            status_rept, msg_rept = self.req_rept.test()
            if status_rept:
                self.logger.debug(f"Received report message from agent with message: {msg_rept}.")

                self.process_msg_rept(msg_rept)
                self.req_rept = self.comm.irecv(tag=TAGS.DATA_RUN_REPORT)

        return

    ################ GENERATION OPERATION ################
    def get_current_generation(self):
        return self.collection_of_generations[-1]

    def check_available_agent(self):
        """Check which agents are available for new jobs"""
        available = []
        for rank, status in self.available_agents.items():
            if status.assigned_job is None:
                available.append(rank)
        return available

    def assign_job(self):
        """Assign jobs from current generation to available agents"""
        cg = self.get_current_generation()
        available_agents = self.check_available_agent()
        
        for agent_rank in available_agents:
            # Check if there are individuals waiting to be distributed
            if not cg.indv_to_distribute:
                break
                
            # Get next individual
            indv = cg.indv_to_distribute.pop(0)
            
            # Check sparsity limitation (same as in Generation.distribute_indv)
            if np.log10(indv.sparsity) >= 1.0:
                # Individual has too high sparsity, assign fake result
                indv('assign', loss=1e9)
                self.logger.info(f'Individual {indv.scope} is killed due to its sparsity = {np.log10(indv.sparsity)} / {indv.sparsity_connection}.')
                continue
            
            # Prepare job data
            job_data = indv('deploy')  # Returns dict with adj_matrix and scope
            
            self.logger.debug(f'Preparing to assign individual {indv.scope} to agent rank {agent_rank}.')
            
            # Prepare message for agent
            msg = {
                'indv_scope': job_data['scope'],
                'graph': self.kwds['agent_behavier']['graph'], # self._adj_matrix_to_graph_string(job_data['adj_matrix']),
                'max_iterations': self.kwds['agent_behavier']['max_iterations'] or 50
            }

            self.logger.debug(f"Prepared job message for individual {indv.scope} to agent rank {agent_rank}: {msg} from kwds: {self.kwds['agent_behavier']}.")
            
            # Send job to agent
            self.comm.isend(msg, dest=agent_rank, tag=TAGS.DATA_ADJ_MATRIX)
            
            # Update agent status
            self.available_agents[agent_rank].assigned_job = job_data['scope']
            
            # Update individual status
            indv.status.assigned.append(agent_rank)
            
            # Add to collection queue
            cg.indv_to_collect.append(indv)
            
            self.logger.info(f'Assigned individual {job_data["scope"]} to agent rank {agent_rank}.')
        
        return

    def _adj_matrix_to_graph_string(self, adj_matrix):
        """Convert adjacency matrix to graph string representation for QCTN"""
        # For now, use a simple string representation
        # In a real implementation, this would convert to the proper quantum circuit graph format
        # that QCTN expects
        return str(adj_matrix.tolist())

    def collect_result(self):
        """Process results from agent_report_buffer and pass to current generation"""
        cg = self.get_current_generation()
        
        while self.agent_report_buffer:
            report = self.agent_report_buffer.popleft()
            
            # Find the individual that matches this report
            indv_scope = self.available_agents[report['rank']].assigned_job
            if indv_scope is None:
                self.logger.warning(f'Received report from agent rank {report["rank"]} but no job was assigned.')
                continue
            
            # Find individual in indv_to_collect
            for indv in cg.indv_to_collect:
                if indv.scope == indv_scope:
                    # Pass result to individual
                    indv('collect', reported_result=report)
                    
                    # Remove agent rank from individual's assigned list
                    if report['rank'] in indv.status.assigned:
                        indv.status.assigned.remove(report['rank'])
                    
                    # Check if individual needs more repetitions
                    indv.status.repeated += 1
                    
                    # Check if individual is finished
                    if indv.status.repeated >= cg.evaluate_repeat:
                        indv.status.finished = True
                        cg.indv_to_collect.remove(indv)
                        self.logger.info(f'Individual {indv.scope} finished after {indv.status.repeated} repetitions.')
                    elif not indv.status.assigned:
                        # Need more repetitions, put back in distribution queue
                        cg.indv_to_collect.remove(indv)
                        cg.indv_to_distribute.append(indv)
                        self.logger.info(f'Individual {indv.scope} needs more repetitions ({indv.status.repeated}/{cg.evaluate_repeat}), re-queuing.')
                    
                    break
        
        return

    def report_agents(self):
        """Report status of all agents"""
        self.logger.info(f'===== AGENT STATUS REPORT (time={self.time}) =====')
        for rank, status in self.available_agents.items():
            self.logger.info(f'Agent rank {rank}: {status}')
        return

    def report_generation(self):
        """Report status of current generation"""
        cg = self.get_current_generation()
        self.logger.info(f'===== GENERATION STATUS REPORT (time={self.time}) =====')
        self.logger.info(f'Generation name: {cg.name}')
        self.logger.info(f'Individuals to distribute: {len(cg.indv_to_distribute)}')
        self.logger.info(f'Individuals to collect: {len(cg.indv_to_collect)}')
        
        # Report societies
        for society_name, society in cg.societies.items():
            finished_count = sum(1 for indv in society.individuals if indv.status.finished)
            self.logger.info(f'Society {society_name}: {finished_count}/{len(society.individuals)} individuals finished')
        
        return

    def span_generation(self):

        self.logger.info(f"Spanning generation at time {self.time}. Current number of generations: {len(self.collection_of_generations)}/{self.max_generation}.")

        if not len(self.collection_of_generations):
            self.collection_of_generations.append(Generation(name='generation_init', **self.kwds))
            return True

        if len(self.collection_of_generations) >= self.max_generation:
            return False
        
        else:
            ## is_finished now is TOTALLY finished, including evaluation and evolution
            current_generation = self.collection_of_generations[-1]
            if current_generation.is_finished():
                next_generation = Generation(current_generation,
                    name=f'generation_{len(self.collection_of_generations)+1:03d}', **self.kwds)
                self.collection_of_generations.append(next_generation)

            return True

    def generate_picklable_instance(self):
        pass

    def __call__(self):
        ## when the overload is called
        ## 1. initilize 4 mpi recv comm
        ## 2. sync goal with the rank = 0
        ## 3. entering the main generation spanning loop,
        ##    different from the former version,
        ##    now the overlord only send the messages from agent to generation,
        ##    the generation will deal with that.
        ## 4. clean comm and send finish msg to all the agents

        self.req_surv = self.comm.irecv(tag=TAGS.INFO_SURVIVAL)
        self.req_estm = self.comm.irecv(tag=TAGS.INFO_TIME_ESTIMATION)
        self.req_abnm = self.comm.irecv(tag=TAGS.INFO_ABNORMAL)
        self.req_rept = self.comm.irecv(tag=TAGS.DATA_RUN_REPORT)

        self.sync_goal()
        
        while self.span_generation():
            cg = self.get_current_generation()
            
            # Periodically check and assign jobs
            self.call_with_interval(self.check_available_agent, 4)()
            self.call_with_interval(self.assign_job, 4)()
            self.call_with_interval(self.collect_result, 4)()
            self.call_with_interval(self.report_agents, 180)()
            self.call_with_interval(self.report_generation, 160)()
            
            # Collect messages from agents
            self.tik_and_collect_everything_from_agent(2)
        else:
            CALLBACKS.OVERLOAD()
            self.host_status = SURVIVAL.HOST_NORMAL_FINISHED
            

        
        self.req_estm.Cancel();self.req_estm.Free()
        self.req_surv.Cancel();self.req_surv.Free()
        self.req_abnm.Cancel();self.req_abnm.Free()
        self.req_rept.Cancel();self.req_rept.Free()
        self.broadcast_finish()
        
        return

if __name__ == '__main__':
    pipeline = MPI_Overlord
    pipeline()
