import time
import tempfile
from pathlib import Path

from tneq_qc.config import Configuration
from tneq_qc.core.qctn import QCTN, QCTNHelper
from tneq_qc.core.cqctn import ContractorQCTN
from tneq_qc.backends.copteinsum import ContractorOptEinsum
from tneq_qc.core.engine import Engine
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

def generate_Mx_phi_x_uniform(num_batch, batch_size, num_qubits, K, edge_size):
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

    weights = init_normalization_factors_vectorized()
    weights = weights[None, None, :K]

    for i in range(num_batch):
        x = torch.empty((batch_size, num_qubits), device='cuda')
        delta = 5 / edge_size
        step = 10 / edge_size
        for dx in range(edge_size):
            for dy in range(edge_size):
                x[dx * edge_size + dy, :] = torch.tensor([dx * step - 5 + delta / 2, dy * step - 5 + delta / 2], device='cuda')        
        
        # print('x', x, x.shape)

        out = eval_hermitenorm_batch(K - 1, x)  # shape = (K, B, D)
        
        # print('out', out.shape)

        out.transpose_(0, 1).transpose_(1, 2)  # shape = (B, D, K)

        # print('out', out, out.shape)
        # print('x', x, x.shape)

        out = weights * torch.sqrt(torch.exp(- torch.square(x) / 2))[:, :, None] * out

        print(f'phi_x.shape {out.shape}')
        # out_norm = torch.sum(out * out, dim=-1)
        # out = out / torch.sqrt(out_norm)[:, :, None]

        # print(f"out after weighting and scaling: {out}, out.shape: {out.shape}")
        einsum_expr = "abc,abd->abcd"
        Mx = torch.einsum(einsum_expr,
                          out, out)
        # print(f"Mx : {Mx}, Mx.shape: {Mx.shape}")
        print(f"Mx.shape: {Mx.shape}")

        Mx_list = [Mx[:, i] for i in range(num_qubits)]
        data_list += [(Mx_list, out)]
    return data_list

def generate_Mx_phi_x_data(num_batch, batch_size, num_qubits, K):
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

    weights = init_normalization_factors_vectorized()
    weights = weights[None, None, :K]

    for i in range(num_batch):
        
        # x变成-5到5的高斯分布
        # x = torch.empty((batch_size, num_qubits), device='cuda').trunc_normal_(mean=0.0, std=1.0, a=-5.0, b=5.0)
        x = torch.empty((batch_size, num_qubits), device='cuda').normal_(mean=0.0, std=1.0)
        
        # print('x', x, x.shape)
        
        out = eval_hermitenorm_batch(K - 1, x)  # shape = (K, B, D)
        
        # print('out', out.shape)

        out.transpose_(0, 1).transpose_(1, 2)  # shape = (B, D, K)

        # print('out', out, out.shape)
        # print('x', x, x.shape)

        out = weights * torch.sqrt(torch.exp(- torch.square(x) / 2))[:, :, None] * out

        print(f'phi_x.shape {out.shape}')
        # out_norm = torch.sum(out * out, dim=-1)
        # out = out / torch.sqrt(out_norm)[:, :, None]

        # print(f"out after weighting and scaling: {out}, out.shape: {out.shape}")
        einsum_expr = "abc,abd->abcd"
        Mx = torch.einsum(einsum_expr,
                          out, out)
        # print(f"Mx : {Mx}, Mx.shape: {Mx.shape}")
        print(f"Mx.shape: {Mx.shape}")

        Mx_list = [Mx[:, i] for i in range(num_qubits)]
        data_list += [(Mx_list, out)]
    return data_list

def generate_circuit_states_list(num_qubits, K):
    """
    Generate circuit states list with status [0, 0, ..., 1] for each qubit
    Parameters:
    - num_qubits: Number of qubits
    - K: Dimension of each qubit state
    Returns:
    - circuit_states_list: List of tensors representing the circuit states for each qubit
    """
    circuit_states_list = [torch.zeros(K, device='cuda') for _ in range(num_qubits)]

    for i in range(len(circuit_states_list)):
        circuit_states_list[i][-1] = 1.0

    return circuit_states_list

def joint_probability_with_heatmap(qctn_cores_file="./assets/qctn_cores.safetensors"):
    backend_type = 'pytorch'

    seed = 42
    # 旧方式：直接使用torch设置随机种子
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    engine = Engine(backend=backend_type)
    
    # 新方式：使用backend的set_random_seed方法
    engine.backend.set_random_seed(seed)

    qctn_graph = QCTNHelper.generate_example_graph()
    print(f"qctn_graph: \n{qctn_graph}")
    # qctn = QCTN(qctn_graph, backend=engine.backend)
    qctn = QCTN.from_pretrained(qctn_graph, qctn_cores_file, backend=engine.backend)

    edge_size = 100

    N = 1
    B = edge_size * edge_size
    D = qctn.nqubits
    K = 3

    data_list = generate_Mx_phi_x_uniform(num_batch=N, batch_size=B, num_qubits=D, K=K, edge_size=edge_size)

    print('data_list[0] shape:', data_list[0][-1].shape)

    data_list = [
        {
            "measure_input_list": x[0],
            # "measure_is_matrix": True,
        } for x in data_list
    ]

    circuit_states_list = generate_circuit_states_list(num_qubits=D, K=K)

    with torch.no_grad():
        result = engine.contract_with_std_graph(
            qctn,
            circuit_states_list=circuit_states_list,
            measure_input_list=data_list[0]["measure_input_list"],
        )

    print(f"Initial contraction result: {result} {result.shape}")

    heatmap = result.reshape(edge_size, edge_size).cpu().numpy()
    import matplotlib.pyplot as plt
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Contraction Result Heatmap')
    plt.savefig('./assets/contraction_result_heatmap.png')


    measure_input_list = data_list[0]["measure_input_list"]

    measure_input_list = [x[::edge_size] for x in measure_input_list]

    measure_input_list[1][:, ] = torch.eye(K, device='cuda')

    print('measure_input_list shape:', [x.shape for x in measure_input_list])
    print(f"first sample {measure_input_list[0][0]}, {measure_input_list[1][0]}")
    print(f"second sample {measure_input_list[0][1]}, {measure_input_list[1][1]}")

    with torch.no_grad():
        result2 = engine.contract_with_std_graph(
            qctn,
            circuit_states_list=circuit_states_list,
            measure_input_list=measure_input_list,
        )
    print(f"Modified contraction result: {result2} {result2.shape}")

    std_result = torch.sum(result.reshape(edge_size, edge_size), axis=-1) / 10.0
    print(f"compare results: ", std_result, std_result.shape)

if __name__ == "__main__":
    joint_probability_with_heatmap(qctn_cores_file="./assets/qctn_cores_2qubits_exp1.safetensors")