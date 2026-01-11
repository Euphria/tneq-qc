import time
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.config import Configuration
from tneq_qc.core.qctn import QCTN, QCTNHelper
from tneq_qc.core.cqctn import ContractorQCTN
from tneq_qc.backends.copteinsum import ContractorOptEinsum
# from tneq_qc.core.engine import Engine
from tneq_qc.core.engine_siamese import EngineSiamese
from tneq_qc.optim.optimizer import Optimizer
import numpy as np
import torch
import math

def generate_circuit_states_list(num_qubits, K, device='cuda'):
    """
    Generate circuit states list with status [0, 0, ..., 1] for each qubit
    Parameters:
    - num_qubits: Number of qubits
    - K: Dimension of each qubit state
    Returns:
    - circuit_states_list: List of tensors representing the circuit states for each qubit
    """
    circuit_states_list = [torch.zeros(K, device=device) for _ in range(num_qubits)]

    for i in range(len(circuit_states_list)):
        circuit_states_list[i][-1] = 1.0

    return circuit_states_list

if __name__ == "__main__":
    backend_type = 'pytorch'

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    backend = BackendFactory.create_backend(backend_type, device='cpu')

    engine = EngineSiamese(backend=backend, strategy_mode="balanced", mx_K=100)

    qctn_graph = QCTNHelper.generate_example_graph()
    # print(f"qctn_graph: \n{qctn_graph}")
    qctn = QCTN(qctn_graph, backend=engine.backend)

    N = 100
    B = 128
    # B = 4
    D = qctn.nqubits
    K = 3
    num_step = 1000

    data_list = []
    
    device = backend.backend_info.device
    
    for i in range(N):
        x = torch.empty((B, D), device=device).normal_(mean=0.0, std=1.0)
        Mx_list, out = engine.generate_data(x, K=K)
        data_list += [({"measure_input_list": Mx_list}, out)]

    # data_list structure is [(dict, out), ...]
    # We only need the dict part for optimizer
    data_list_for_optim = [x[0] for x in data_list]

    circuit_states_list = generate_circuit_states_list(num_qubits=D, K=K, device=backend.backend_info.device)

    # torch profiler memory usage test
    import torch.profiler

    # torch.cuda.empty_cache()
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     schedule=torch.profiler.schedule(
    #         wait=1,
    #         warmup=1,
    #         active=3),
    #     # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/tneq_qc_run'),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # ) as prof:
    #     for step in range(100):
    #         with torch.no_grad():
    #             result = engine.contract_with_self(qctn, 
    #                                                 circuit_states=circuit_states_list,
    #                                                 measure_input=data_list[0]["measure_input_list"], 
    #                                                 measure_is_matrix=True,
    #                                                 )
    #         prof.step()
    # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    # torch.cuda.empty_cache()
    # with torch.profiler.profile(
    #     activities=[
    #         # torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     schedule=torch.profiler.schedule(
    #         wait=1,
    #         warmup=1,
    #         active=3),
    #     # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/tneq_qc_run'),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # ) as prof:
    #     for step in range(10):
    #         with torch.no_grad():
    #             result = engine.contract_with_std_graph(qctn,
    #                                                     circuit_states_list=circuit_states_list,
    #                                                     measure_input_list=data_list[0]["measure_input_list"],
    #                                                     )
    #         prof.step()
    # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    # torch.cuda.empty_cache()
    # with torch.no_grad():
    #     result = engine.contract_with_self(qctn, 
    #                                         circuit_states=circuit_states_list,
    #                                         measure_input=data_list[0]["measure_input_list"], 
    #                                         measure_is_matrix=True,
    #                                         )
    # print(f"Initial Result: {[result[x].item() for x in range(10)]}")

    # print(f"已分配显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    # print(f"缓存显存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # exit()

    # torch.cuda.empty_cache()
    # with torch.no_grad():
    #     # for i in range(10):
    #     result = engine.contract_with_std_graph(qctn,
    #                                             circuit_states_list=circuit_states_list,
    #                                             measure_input_list=data_list[0]["measure_input_list"],
    #                                             )
    # print(f"Initial Result (std graph): {[result[x].item() for x in range(10)]}")

    # print(f"已分配显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    # print(f"缓存显存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # exit()

    # Define step-based learning rate schedule
    # Format: list of (step, lr) tuples
    # At step 0: lr = 1e-2
    # At step 200: lr drops to 1e-3 (10x decay)
    # At step 800: lr drops to 1e-4 (another 10x decay)
    lr_schedule = [
        (0, 1e-2),
        (200, 5e-3),
        (600, 2e-3),
        (800, 1e-3),
    ]

    optimizer = Optimizer(
        method='sgdg', 
        max_iter=num_step, 
        # tol=1e-6, 
        tol=0.0, 
        learning_rate=1e-2, 
        beta1=0.9, 
        beta2=0.95, 
        epsilon=1e-8,
        engine=engine,
        lr_schedule=lr_schedule,

        momentum=0.9,            # 动量因子
        stiefel=True,            # 启用 Stiefel 流形优化
    )
    
    torch.cuda.empty_cache()
    
    tic = time.time()

    optimizer.optimize(qctn, 
                       data_list=data_list_for_optim, 
                    #    circuit_states=circuit_states_list,
                       circuit_states_list=circuit_states_list,
                       )

    toc = time.time()
    
    print(f"已分配显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"缓存显存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Optimization Time: {toc - tic:.2f} seconds")

    # save cores
    cores_file = "./assets/qctn_cores.safetensors"
    qctn.save_cores(cores_file, metadata={"graph": "example"})

    exit()

    # Choose the first data of each batch for testing
    test_loss_list = []
    for i in range(N):
        data_slice = [x[0:1] for x in data_list[i]["measure_input_list"]]
        result = engine.contract_with_std_graph(qctn, 
                                                circuit_states_list=circuit_states_list,
                                                measure_input_list=data_slice, 
                                                )
        print(f"Test {i}, Result: {result.item()}")
        test_loss_list.append(result.item())

    print(f"Average Result: {sum(test_loss_list) / len(test_loss_list)}")
    print(f"Max Result: {max(test_loss_list)}")
    print(f"Min Result: {min(test_loss_list)}")

    
    # load pretrained qctn
    pretrained_qctn = QCTN.from_pretrained(qctn_graph, cores_file, backend=engine.backend)
    with torch.no_grad():
        pretrained_result = engine.contract_with_std_graph(
            pretrained_qctn,
            circuit_states_list=circuit_states_list,
            measure_input_list=data_list[0]["measure_input_list"],
        )
    print(f"Pretrained Result (std graph): {pretrained_result} {pretrained_result.shape}")

    # get cross entropy loss for all pretrained results (shape = 512)
    loss = - torch.log(pretrained_result)
    print(f"Cross Entropy Loss: {loss} {torch.mean(loss)}")
