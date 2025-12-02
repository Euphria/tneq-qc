from typing import Any
import numpy as np, os, sys, time, gc, logging
np.set_printoptions(precision=2)
import jax
import jax.numpy as jnp
import mpi4py
from .mpi_core import TAGS, REASONS, SURVIVAL, load_func
from ..core.qctn import QCTN
from ..core.cqctn import ContractorQCTN
from ..backends.copteinsum import ContractorOptEinsum
from ..config import Configuration

class MPI_Agent(object):

    ## AGENT SEND:
    ## 1. survival info
    ## 2. abnormal when receiving job
    ## 3. estimation info
    ## 4. job result report

    ## AGENT RECEIVE
    ## 1. survival ping
    ## 2. job data 

    ################ UTILS ################
    def __init__(self, comm: mpi4py.MPI.COMM_WORLD, **kwargs) -> None:
        self.kwargs = kwargs
        self.comm = comm
        self.rank = comm.Get_rank()
        self.time = 0
        self.busy_status = False
        self.start_time = time.time()
        self.logger = kwargs['logger']
        self.logger.info(f'MPI_Agent {self.rank} started.')
        self.optimizer = load_func(kwargs['optimization']['optimizer'])
        self.optimizer_param = kwargs['optimization']['optimizer_params']
        
        # QCTN target is passed in during initialization (already initialized)
        self.qctn_target = None  # Will be set in sync_goal
        
        # Determine contraction engine
        self.contraction_engine = ContractorOptEinsum if Configuration.contraction_engine == 'opt_einsum' else ContractorQCTN

    def tik(self, sec):
        self.time += sec
        time.sleep(sec)

    ################ MPI COMMUNICATION ################
    def sync_goal(self):
        """Synchronize the target QCTN from Overlord (already initialized)"""
        try:
            self.qctn_target = None
            self.qctn_target = self.comm.bcast(self.qctn_target, root=0)
            self.logger.info(f'Received qctn_target with {self.qctn_target.nqubits} qubits and {self.qctn_target.ncores} cores.')
        except:
            self.logger.error(f'Agent {self.rank} reported errors in receiving qctn_target')
            raise

    def report_surival(self, current_iter, max_iter):
        status, msg = self.req_surv.test()

        self.logger.debug(f'Agent {self.rank} checking for survival ping: status={status}, msg={msg}')

        if status:
            self.req_surv = self.comm.irecv(source=0, tag=TAGS.INFO_SURVIVAL)

            if msg:
                return msg
            real_up_time = time.time() - self.start_time
            return_dict = {
                'rank': self.rank,
                'time': self.time,
                'real_up_time': real_up_time,
                'busy': self.busy_status,
                'current_iter': current_iter,
                'max_iter': max_iter
            }
            self.comm.isend(return_dict, dest=0, tag=TAGS.INFO_SURVIVAL)
            if self.busy_status:
                self.logger.info(f'Received survival test signal {SURVIVAL.__rdict__[msg]} from overlord, ' \
                                f'reported tik time {self.time}, real up time {real_up_time},' \
                                f'current completion rate {current_iter} / {max_iter}.')
            else:
                self.logger.info(f'Received survival test signal {SURVIVAL.__rdict__[msg]} from overlord, ' \
                                f'reported tik time {self.time}, real up time {real_up_time},' \
                                f'not working currently.')
                
        return msg

    def receive_job(self):
        status, msg = self.req_adjm.test()
        
        self.logger.debug(f'Agent {self.rank} checking for job: status={status}, msg={msg}')

        if status:
            self.req_adjm = self.comm.irecv(source=0, tag=TAGS.DATA_ADJ_MATRIX)

            try:
                job = self.prepare_job(msg)
                
                return job
            except Exception as e:
                self.logger.error(f'Agent {self.rank} failed to prepare job from message: {msg}, error: {e}', exc_info=True)

                self.req_adjm.Cancel();self.req_adjm.Free()
                # self.req_adjm = self.comm.irecv(source=0, tag=TAGS.DATA_ADJ_MATRIX)
                self.comm.isend(self.rank, dest=0, tag=TAGS.INFO_ABNORMAL)

                # TODO: process job failure here and outter
                return None
        else:
            self.logger.debug(f"No job received yet by agent {self.rank}. req_adjm: {self.req_adjm}")
            return None

    def report_estimation(self, return_dict):
        self.comm.isend(return_dict, dest=0, tag=TAGS.INFO_TIME_ESTIMATION)
        return
    
    def report_result(self, report):
        self.comm.isend(report, dest=0, tag=TAGS.DATA_RUN_REPORT)
        self.busy_status = False
        return
    
    ################ DO JOB WITH QCTN ################
    def prepare_job(self, msg):
        """
        Prepare QCTN job from received message.
        
        Args:
            msg: Dictionary containing:
                - 'indv_scope': Individual identifier
                - 'graph': Quantum circuit graph string
                - 'max_iterations': Maximum optimization iterations
        
        Returns:
            tuple: (qctn_example, max_iterations, optimizer_instance)
        """
        indv_scope = msg['indv_scope']
        graph = msg['graph']  # Quantum circuit graph string
        max_iterations = msg['max_iterations']

        self.busy_status = True
        
        # Build and initialize QCTN from the graph
        qctn_example = QCTN(graph)
        
        # Create optimizer instance
        optimizer_instance = self.optimizer(**self.optimizer_param)
        optimizer_instance.max_iter = max_iterations
        
        self.logger.info(f'Received job indv {indv_scope} from overlord, ' \
                         f'qctn_example has {qctn_example.nqubits} qubits and {qctn_example.ncores} cores, '
                         f'gonna run {max_iterations} iterations.')

        return qctn_example, max_iterations, optimizer_instance, indv_scope
    
    def do_estimation(self, qctn_example, required_time):
        """Calculate current loss and estimate remaining time"""
        loss, _ = qctn_example.contract_with_QCTN_for_gradient(self.qctn_target, engine=self.contraction_engine)
        loss = float(loss)  # Convert JAX array to Python float

        return_dict = {
            'rank': self.rank,
            'loss': loss,
            'required_time': required_time,
        }
        self.logger.info(f'Reporting estimation time {required_time} with current loss {loss}.')

        return return_dict

    def prepare_report_result(self, qctn_example, current_iter, reason, scope):
        """Prepare final result report and clean up resources"""
        
        loss, _ = qctn_example.contract_with_QCTN_for_gradient(self.qctn_target, engine=self.contraction_engine)
        loss = float(loss)  # Convert JAX array to Python float
        
        # Clean up QCTN resources
        del qctn_example
        gc.collect()

        return_dict = {
            'rank': self.rank,
            'loss': loss,
            'current_iter': current_iter,
            'reason': reason,
            'indv_scope': scope,
        }
        self.logger.info(f'Reporting result {loss} at iteration {current_iter} with reason {reason}.')

        return return_dict

    def evaluate(self, qctn_example, optimizer_instance, n_iter) -> None:
        """
        Execute optimization iterations.
        
        Args:
            qctn_example: QCTN instance to optimize
            optimizer_instance: Optimizer instance
            n_iter: Number of iterations to run
        """
        for i in range(n_iter):
            loss, grads = qctn_example.contract_with_QCTN_for_gradient(self.qctn_target, engine=self.contraction_engine)
            
            # Check convergence
            if loss < optimizer_instance.tol:
                self.logger.debug(f"Convergence achieved at iteration {optimizer_instance.iter} with loss {loss}.")
                break
            
            # Update parameters using optimizer step
            optimizer_instance.step(qctn_example, grads)
            optimizer_instance.iter += 1
            
        return 

    ################ ENTRANCE ################

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        ## when the agent is called
        ## 1. initilize two mpi recv comm
        ## 2. sync goal with the rank = 0
        ## 3. entering the main while True loop
        ##      i. check if need to report surival state
        ##      ii. try to receive a job, initilized the tf sess and obtain a step
        ##      iii. run first N step, estimate overall run time, report to overload
        ##      iv. if estimate time is largely overhaul, then finish this run and report failure
        ##      v. run N step, try report surival state and count time
        ##      vi. if reach timeout or max step, stop run and exit while True loop
        ##      vii. report current steps and loss, clean tf sess and graph
        ##      viii. repeat from i. until the overload tell all things done
        ## 4. clean comm and report finish to rank = 0

        self.req_adjm = self.comm.irecv(source=0, tag=TAGS.DATA_ADJ_MATRIX)
        self.req_surv = self.comm.irecv(source=0, tag=TAGS.INFO_SURVIVAL)
        self.sync_goal()

        call_start_time = time.time()
        timeout = self.kwargs['agent_behavier']['timeout']
        n_iter = self.kwargs['agent_behavier']['n_iter']
        estimation_iter = self.kwargs['agent_behavier']['estimation_iter']

        if estimation_iter % n_iter:
            estimation_iter = int(estimation_iter/n_iter) * estimation_iter

        allow_waiting_after_timeout_rate = self.kwargs['agent_behavier']['allow_waiting_after_timeout_rate']

        current_iter, job, qctn_example, max_iterations, optimizer_instance = None, None, None, None, None
        while True:
            msg = self.report_surival(current_iter, max_iterations)
            if msg:
                self.logger.info(f'Received signal {SURVIVAL.__rdict__[msg]} from host, breaking from while loop.')
                break

            # waiting for job
            if not self.busy_status:
                job = self.receive_job()
                if not job:
                    self.tik(1)
                    continue
                else:
                    qctn_example, max_iterations, optimizer_instance, indv_scope = job
                    self.busy_status = True
                    current_iter = 0

            self.logger.debug(f'Agent {self.rank} entering evaluation loop at iteration {current_iter} / {max_iterations}.')

            # received job, everything is fine
            if current_iter < max_iterations:

                if current_iter == estimation_iter:
                    required_time = (max_iterations / estimation_iter) * (time.time() - call_start_time)
                    estimatation_report = self.do_estimation(qctn_example, required_time)
                    self.report_estimation(estimatation_report)

                # in time
                if time.time() - call_start_time < timeout:
                    self.evaluate(qctn_example, optimizer_instance, n_iter)
                    current_iter += n_iter  # Update iteration counter

                # timeout
                else:
                    # wait until finish when it is bearable
                    if current_iter / max_iterations > allow_waiting_after_timeout_rate:
                        self.evaluate(qctn_example, optimizer_instance, n_iter)
                        current_iter += n_iter  # Update iteration counter
                    
                    # shutdown the computation
                    else:
                        report = self.prepare_report_result(qctn_example, current_iter, REASONS.HARD_TIMEOUT, indv_scope)
                        self.report_result(report)
                        current_iter = max_iterations  # Mark as done to exit loop
            
            else:
                report = self.prepare_report_result(qctn_example, current_iter, REASONS.REACH_MAX_ITER, indv_scope)
                self.report_result(report)
                current_iter = None  # Reset for next job
                self.busy_status = False

        self.req_adjm.Cancel();self.req_adjm.Free()
        self.req_surv.Cancel();self.req_surv.Free()
        self.logger.info(f'MPI_Agent {self.rank} finished.')

        return
