#!/usr/bin/env python3
"""
Test script for MPI_Overlord class
This script tests the MPI_Overlord with multiple agents

Usage:
    mpiexec -n 4 python test_mpi_overlord.py
    (1 master/overlord + 3 agents)
"""

import sys
import os
import logging
import numpy as np
from mpi4py import MPI

# Add parent directory to path
if __package__ is None or __package__ == '':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tneq_qc.distributed.mpi_overlord import MPI_Overlord
from tneq_qc.distributed.mpi_agent import MPI_Agent
from tneq_qc.core.qctn import QCTN, QCTNHelper
from tneq_qc.log_utils import setup_colored_logger


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Setup logger
    logger = setup_colored_logger("MPI_Test", rank, level=logging.DEBUG)
    
    if size < 2:
        if rank == 0:
            logger.error("This test requires at least 2 processes (1 overlord + 1 agent)")
            logger.error("Usage: mpiexec -n 4 python test_mpi_overlord.py")
        sys.exit(1)
    
    logger.info(f"Starting test with {size} processes ({size-1} agents)")
    
    # Configuration dictionary
    config = {
        'logger': logger,
        'experiment': {
            'max_generation': 2,  # Run 2 generations for testing
            'evoluation_goal': None,  # Will be created and broadcast
        },
        'generation_property': {
            'evaluate_repeat': 2,  # Each individual evaluated 2 times
            'still_allow_repeat_after_hard_timeout': True,
            'n_societies': 1,
            'society_property': {
                'society': [
                    {
                        'n_individuals_span': 4,  # Small number for testing
                        'n_individuals_survive': 2,
                        'fitness_func': lambda s, l: s + 100 * l
                    }
                ]
            }
        },
        'individual_property': {
            'discard_hard_timeout_result': False,
            'random_initilization_property': {
                'tn_size': 4,
                'tn_rank': 2,
                'presented_shape': 2,
                'init_sparsity': 0.3
            }
        },
        'agent_behavier': {
            'timeout': 300,  # 5 minutes timeout
            'n_iter': 10,    # Run 10 iterations at a time
            'estimation_iter': 20,  # Estimate after 20 iterations
            'max_iterations': 100,  # Total 100 iterations per individual
            'allow_waiting_after_timeout_rate': 0.8,
            'graph': QCTNHelper.generate_example_graph(target=False)
        },
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
        'evolution_property': {
            'elimiation_threshold': [0.5],  # Eliminate bottom 50%
            'immigration_prob': 0.1,
            'immigration_number': 1,
            'crossover_alpha': 0.5,
            'mutation_prob': 0.1
        }
    }
    
    if rank == 0:
        # Master process - run Overlord
        logger.info("="*60)
        logger.info("MASTER PROCESS - Running MPI_Overlord")
        logger.info("="*60)
        
        # Create test QCTN as evaluation goal
        target_graph = QCTNHelper.generate_example_graph(target=True)
        qctn_target = QCTN(target_graph)
        logger.info(f"Created test QCTN with {qctn_target.nqubits} qubits and {qctn_target.ncores} cores")
        
        # Save to temporary file for broadcast
        temp_goal_path = '/tmp/test_qctn_goal.npy'
        np.save(temp_goal_path, qctn_target)
        config['experiment']['evoluation_goal'] = temp_goal_path
        
        # Instead of broadcasting numpy array, we'll broadcast the QCTN object directly
        # The overlord's sync_goal needs to be updated to handle QCTN objects
        
        try:
            overlord = MPI_Overlord(comm, **config)
            
            # Directly set the QCTN target (bypass file loading for this test)
            overlord.evoluation_goal = qctn_target
            
            # Call overlord
            overlord()
            
            logger.info("="*60)
            logger.info("OVERLORD FINISHED SUCCESSFULLY")
            logger.info("="*60)
            
            # Print final results
            for gen_idx, generation in enumerate(overlord.collection_of_generations):
                logger.info(f"\n===== Generation {gen_idx}: {generation.name} =====")
                for society_name, society in generation.societies.items():
                    logger.info(f"\nSociety {society_name}:")
                    for indv in society.individuals:
                        if indv.status.finished:
                            logger.info(f"  {indv.scope}: "
                                      f"sparsity={indv.sparsity:.3f}, "
                                      f"losses={[f'{l:.4f}' for l in indv.repeat_loss]}, "
                                      f"iterations={indv.repeat_loss_iter}")
                
        except Exception as e:
            logger.error(f"Overlord encountered error: {e}", exc_info=True)
            raise
        finally:
            # Cleanup temp file
            if os.path.exists(temp_goal_path):
                os.remove(temp_goal_path)
    
    else:
        # Worker process - run Agent
        logger.info("="*60)
        logger.info(f"WORKER PROCESS - Running MPI_Agent")
        logger.info("="*60)
        
        try:
            agent = MPI_Agent(comm, **config)
            agent()
            
            logger.info("="*60)
            logger.info(f"AGENT FINISHED SUCCESSFULLY")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Agent encountered error: {e}", exc_info=True)
            raise
    
    comm.Barrier()
    
    if rank == 0:
        logger.info("\n" + "="*60)
        logger.info("ALL PROCESSES COMPLETED SUCCESSFULLY")
        logger.info("="*60)


if __name__ == '__main__':
    main()
