from typing import Any
import numpy as np, os, sys, time, gc
import logging
from mpi4py import MPI
from tneq_qc.distributed.mpi_agent import MPI_Agent
from tneq_qc.distributed.mpi_core import TAGS, REASONS, SURVIVAL
from tneq_qc.core.qctn import QCTNHelper, QCTN
from tneq_qc.optim.optimizer import Optimizer
from tneq_qc.log_utils import setup_colored_logger


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    if rank == 0:
        print(f"Error: This test requires exactly 2 MPI processes (1 master + 1 worker), got {size}")
        print("Run with: mpiexec -n 2 python test_mpi_agent.py")
    sys.exit(1)

# Setup colored logger using the helper function
logger = setup_colored_logger(__name__, rank, level=logging.DEBUG)

# Configuration for agent
kwargs = {
    'logger': logger,
    'optimization': {
        'optimizer': 'tneq_qc.optimizer.Optimizer',  # Will be loaded by load_func
        'optimizer_params': {
            'method': 'adam',
            'learning_rate': 0.1,
            'max_iter': 100,
            'tol': 1e-6,
            'beta1': 0.9,
            'beta2': 0.95,
            'epsilon': 1e-8
        }
    },
    'agent_behavier': {
        'timeout': 300,  # 5 minutes
        'n_iter': 10,    # Run 10 iterations at a time
        'estimation_iter': 20,  # Estimate time after 20 iterations
        'allow_waiting_after_timeout_rate': 0.8
    }
}

if rank == 0:
    # ============ MASTER PROCESS ============
    logger.info("Master process started")
    
    # Create target QCTN
    target_graph = QCTNHelper.generate_example_graph(target=True)
    qctn_target = QCTN(target_graph)
    logger.info(f"Created qctn_target with {qctn_target.nqubits} qubits and {qctn_target.ncores} cores")
    
    # Broadcast target QCTN to worker
    logger.info("Broadcasting qctn_target to worker...")
    qctn_target = comm.bcast(qctn_target, root=0)
    
    # Generate example graph for the task
    example_graph = QCTNHelper.generate_example_graph(target=False)
    logger.info(f"Generated example graph:\n{example_graph}")
    
    # Prepare task message
    task_msg = {
        'indv_scope': 'test_individual_001',
        'graph': example_graph,
        'max_iterations': 50
    }
    
    # Send task to worker
    logger.info("Sending task to worker (rank=1)...")
    for i in range(5):
        comm.isend(task_msg, dest=1, tag=TAGS.DATA_ADJ_MATRIX)
    
    # Wait for worker to process
    logger.info("Waiting for worker response...")
    
    # Receive time estimation report
    status = MPI.Status()
    estimation_report = comm.recv(source=1, tag=TAGS.INFO_TIME_ESTIMATION, status=status)
    logger.info(f"Received estimation report: loss={estimation_report['loss']:.6f}, "
                f"estimated_time={estimation_report['required_time']:.2f}s")
    
    # Receive final result
    final_report = comm.recv(source=1, tag=TAGS.DATA_RUN_REPORT, status=status)
    logger.info(f"Received final report: loss={final_report['loss']:.6f}, "
                f"iterations={final_report['current_iter']}, "
                f"reason={REASONS.__rdict__[final_report['reason']]}")
    
    # Send termination signal
    logger.info("Sending termination signal to worker...")
    comm.isend(SURVIVAL.HOST_NORMAL_FINISHED, dest=1, tag=TAGS.INFO_SURVIVAL)
    
    logger.info("Master process finished successfully")
    
else:
    # ============ WORKER PROCESS ============
    logger.info("Worker process started")
    
    # Create MPI_Agent instance
    agent = MPI_Agent(comm, **kwargs)
    
    # Call the agent - this runs the complete worker loop
    # The __call__ method handles:
    # - Initializing irecv requests (req_adjm, req_surv)
    # - Syncing qctn_target from master
    # - Waiting for and processing jobs
    # - Running optimization iterations
    # - Reporting estimation and final results
    # - Handling termination signals
    # - Cleaning up MPI resources
    agent()
    
    logger.info("Worker process finished successfully")