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

def init_normalization_factors_vectorized(k_max=100, device='cuda'):
    """
    向量化计算
    """
    # 在 CPU 上计算（因为 torch.lgamma 在某些版本可能不支持 CUDA）
    k = torch.arange(k_max + 1, dtype=torch.float32)
    
    # 使用 lgamma 计算 log(k!)
    # lgamma(k+1) = log(k!)
    log_factorial = torch.lgamma(k + 1)
    
    log_2pi = math.log(2 * math.pi)
    log_factor = -0.5 * (0.5 * log_2pi + log_factorial)
    
    return torch.exp(log_factor).to(device)

def eval_hermitenorm_batch(n_max, x, device='cuda'):
    """
    一次性计算从 0 到 n_max 的所有 Hermite 多项式
    
    返回: shape = (n_max+1, *x.shape)
    """
    x = torch.tensor(x, dtype=torch.float32, device=device) if not isinstance(x, torch.Tensor) else x.to(device)
    
    H = torch.zeros((n_max + 1,) + x.shape, dtype=x.dtype, device=device)
    H[0] = torch.ones_like(x)
    
    if n_max >= 1:
        # H[1] = 2 * x
        H[1] = x
        
        for i in range(2, n_max + 1):
            H[i] = x * H[i-1] - (i-1) * H[i-2]
    
    return H

def generate_Mx_phi_x_data(num_batch, batch_size, num_qubits, K, device='cuda'):
    """
    Generate Mx and phi_x data
    Parameters:
    - num_batch: Number of batches to generate
    - batch_size: Size of each batch
    - num_qubits: Number of qubits (dimension D)
    - K: Number of Hermite polynomials to evaluate
    Returns:
    - data_list: List of tuples (Mx, phi_x) for each batch
    """

    data_list = []

    weights = init_normalization_factors_vectorized(device=device)
    weights = weights[None, None, :K]

    for i in range(num_batch):
        
        # x变成-5到5的高斯分布
        # x = torch.empty((batch_size, num_qubits), device='cuda').uniform_(-5.0, 5.0)
        x = torch.empty((batch_size, num_qubits), device=device).normal_(mean=2.5, std=1.0)
        
        # print('x', x, x.shape)
        
        out = eval_hermitenorm_batch(K - 1, x, device=device)  # shape = (K, B, D)
        
        # print('out', out.shape)

        out.transpose_(0, 1).transpose_(1, 2)  # shape = (B, D, K)

        # print('out', out, out.shape)
        # print('x', x, x.shape)

        out = weights * torch.sqrt(torch.exp(- torch.square(x) / 2))[:, :, None] * out

        # print(f'phi_x.shape {out.shape}')

        # out_norm = torch.sum(out * out, dim=-1)
        # out = out / torch.sqrt(out_norm)[:, :, None]

        # print(f"out after weighting and scaling: {out}, out.shape: {out.shape}")
        einsum_expr = "abc,abd->abcd"
        Mx = torch.einsum(einsum_expr,
                          out, out)
        # print(f"Mx : {Mx}, Mx.shape: {Mx.shape}")

        # print(f"Mx.shape: {Mx.shape}")

        Mx_list = [Mx[:, i] for i in range(num_qubits)]
        data_list += [(Mx_list, out)]
    return data_list

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

    engine = EngineSiamese(backend=backend, strategy_mode="balanced")

    qctn_graph = QCTNHelper.generate_example_graph()
    # print(f"qctn_graph: \n{qctn_graph}")
    qctn = QCTN(qctn_graph, backend=engine.backend)

    N = 100
    B = 128
    D = qctn.nqubits
    K = 3

    data_list = generate_Mx_phi_x_data(num_batch=N, batch_size=B, num_qubits=D, K=K, device=backend.backend_info.device)

    # print('data_list[0] shape:', data_list[0][-1].shape)

    # for i in range(B):
    #     phi_x = data_list[0][-1][i]
    #     for j in range(D):
    #         phi_x_j = phi_x[j]
    #         norm = torch.sum(phi_x_j * phi_x_j)
    #         print(f"Batch {i}, Qubit {j}, Norm of phi_x_j: {norm.item()}")
    #     print(f"phi_x for batch {i}: {phi_x} shape: {phi_x.shape}")

    data_list = [
        {
            "measure_input_list": x[0], 
            # "measure_is_matrix": True,
        } for x in data_list
    ]

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
        max_iter=1000, 
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
                       data_list=data_list, 
                    #    circuit_states=circuit_states_list,
                       circuit_states_list=circuit_states_list,
                       )

    toc = time.time()
    
    print(f"已分配显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"缓存显存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Optimization Time: {toc - tic:.2f} seconds")

    from pathlib import Path

    # save cores
    cores_file = Path("./assets/qctn_cores.safetensors")

    # 检查并创建目录
    if not cores_file.parent.exists():
        cores_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"首次创建cores file保存目录: {cores_file.parent.absolute()}")

    qctn.save_cores(str(cores_file), metadata={"graph": "example"})
    print(f"cores file保存到: {cores_file.absolute()}")

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
