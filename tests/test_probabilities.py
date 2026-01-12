from tneq_qc.backends import backend_factory
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os

from tneq_qc.core.engine_siamese import EngineSiamese
from tneq_qc.core.qctn import QCTN, QCTNHelper

# ==========================================
# Helper Functions
# ==========================================

def generate_circuit_states_list(num_qubits, K, device='cuda'):
    circuit_states_list = [torch.zeros(K, device=device) for _ in range(num_qubits)]
    for i in range(len(circuit_states_list)):
        circuit_states_list[i][-1] = 1.0
    return circuit_states_list

# ==========================================
# Test Functions
# ==========================================

def test_probabilities():
    print("\n=== Test 0: Simple Probabilities (v2) ===")
    # Setup
    backend_type = 'pytorch'
    backend = backend_factory.BackendFactory.create_backend(backend_type, device='cuda')
    engine = EngineSiamese(backend=backend)
    
    # Create a simple 2-qubit circuit
    graph = "-2-A-2-\n-2-B-2-"
    qctn = QCTN(graph, backend=engine.backend)
    
    # Circuit states: |0> for all qubits
    # shape (B, 2)
    batch_size = 4
    state_0 = torch.tensor([1.0, 0.0], dtype=torch.float32)
    state_0_batch = state_0.unsqueeze(0).expand(batch_size, -1) # (B, 2)
    circuit_states = [state_0_batch, state_0_batch]
    
    circuit_states = [engine.backend.convert_to_tensor(s) for s in circuit_states]

    # Create Projectors
    # |0><0|
    proj_0 = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    proj_0_batch = proj_0.unsqueeze(0).expand(batch_size, -1, -1) # (B, 2, 2)
    
    # 1. Test Full Probability P(00)
    measure_list_full = [proj_0_batch, proj_0_batch]

    measure_list_full = [engine.backend.convert_to_tensor(m) for m in measure_list_full]

    prob_00 = engine.calculate_full_probability(qctn, circuit_states, measure_list_full)
    print(f"P(00) shape: {prob_00.shape}")
    print(f"P(00): {prob_00[0]}")
    
    # 2. Test Marginal Probability P(q0=0)
    measure_list_marginal = [proj_0_batch]

    measure_list_marginal = [engine.backend.convert_to_tensor(m) for m in measure_list_marginal]

    prob_q0_0 = engine.calculate_marginal_probability(qctn, circuit_states, measure_list_marginal, [0])
    print(f"P(q0=0) shape: {prob_q0_0.shape}")
    print(f"P(q0=0): {prob_q0_0[0]}")
    
    measure_list_conditional = [proj_0_batch, proj_0_batch]

    measure_list_conditional = [engine.backend.convert_to_tensor(m) for m in measure_list_conditional]

    # 3. Test Conditional Probability P(q1=0 | q0=0)
    cond_prob = engine.calculate_conditional_probability(
        qctn, 
        circuit_states, 
        measure_input_list=measure_list_conditional, 
        qubit_indices=[0, 1], 
        target_indices=[1]
    )
    print(f"P(q1=0 | q0=0) shape: {cond_prob.shape}")
    print(f"P(q1=0 | q0=0): {cond_prob[0]}")
    
    # Expected: P(00) / P(q0=0)
    expected = prob_00 / (prob_q0_0 + 1e-10)
    print(f"Expected: {expected[0]}")
    
    assert torch.allclose(cond_prob, expected, atol=1e-5)
    print("Conditional probability test passed!")

def test_random_probabilities(qctn_cores_file="./assets/qctn_cores_3qubits_exp1.safetensors", device='cuda'):
    print("\n=== Test 1: Random Probabilities ===")
    backend_type = 'pytorch'
    backend = backend_factory.BackendFactory.create_backend(backend_type, device=device)
    engine = EngineSiamese(backend=backend)
    
    # Load QCTN
    if not os.path.exists(qctn_cores_file):
        print(f"Warning: {qctn_cores_file} not found. Using random initialization.")
        qctn_graph = QCTNHelper.generate_example_graph()
        qctn = QCTN(qctn_graph, backend=engine.backend)
    else:
        qctn_graph = QCTNHelper.generate_example_graph()
        qctn = QCTN.from_pretrained(qctn_graph, qctn_cores_file, backend=engine.backend)
    
    D = qctn.nqubits
    K = 3 # Dimension of state
    
    # Generate Data
    # batch_size=1, num_sample=1 (num_batch=1)
    
    # data = generate_Mx_phi_x_data(num_batch=1, batch_size=1, num_qubits=D, K=K, device=device)
    # measure_input_list = data[0][0] # List of Mx tensors
    
    x = torch.empty((1, D), device=device).normal_(mean=0.0, std=1.0)
    Mx_list, out = engine.generate_data(x, K=K)
    measure_input_list = Mx_list
    
    circuit_states = generate_circuit_states_list(D, K, device=device)
    
    circuit_states = [engine.backend.convert_to_tensor(s) for s in circuit_states]
    measure_input_list = [engine.backend.convert_to_tensor(m) for m in measure_input_list]

    # 1. Full Probability
    prob_full = engine.calculate_full_probability(qctn, circuit_states, measure_input_list)
    print(f"Full Probability: {prob_full.item()}")
    
    # tmp = measure_input_list[0]
    # measure_input_list[0] = None
    # prob_matrix = engine.contract_with_compiled_strategy(qctn, circuit_states, measure_input_list)
    # print(f"prob_matrix: {prob_matrix} {prob_matrix.shape}")

    # print(f"tmp shape: {tmp.shape}")

    # result = torch.einsum("aij,aji->a", tmp, prob_matrix)
    
    # print(f"Recomputed Full Probability: {result.item()}")

    # 2. Marginal Probability for each qubit
    print("\nMarginal Probabilities:")
    for i in range(D):
        # measure_input_list[i] is (B, K, K)
        prob_marg = engine.calculate_marginal_probability(
            qctn, 
            circuit_states, 
            [measure_input_list[i]], 
            [i]
        )
        print(f"Qubit {i}: {prob_marg.item()}")
        
    # 3. Conditional Probabilities
    print("\nConditional Probabilities:")
    
    if D >= 2:
        # Case 1: P(q0 | q1)
        prob_cond_1 = engine.calculate_conditional_probability(
            qctn, circuit_states, 
            [measure_input_list[0], measure_input_list[1]], 
            [0, 1], 
            [0]
        )
        print(f"P(q0 | q1): {prob_cond_1.item()}")
        
        # Case 2: P(q1 | q0)
        prob_cond_2 = engine.calculate_conditional_probability(
            qctn, circuit_states, 
            [measure_input_list[0], measure_input_list[1]], 
            [0, 1], 
            [1]
        )
        print(f"P(q1 | q0): {prob_cond_2.item()}")
        
        # Case 3: P(q0 | q1) with same inputs (sanity check)
        prob_cond_3 = engine.calculate_conditional_probability(
            qctn, circuit_states, 
            [measure_input_list[0], measure_input_list[1]], 
            [0, 1], 
            [0]
        )
        print(f"P(q0 | q1) (repeat): {prob_cond_3.item()}")

        # Case 4: P(q1 | q0) with same inputs (sanity check)
        prob_cond_4 = engine.calculate_conditional_probability(
            qctn, circuit_states, 
            [measure_input_list[0], measure_input_list[1]], 
            [0, 1], 
            [1]
        )
        print(f"P(q1 | q0) (repeat): {prob_cond_4.item()}")

        # Case 5: If D > 2, we could do more. For D=2, we are limited.
        # Let's try P(q0 | q1) but using a subset of measurements?
        # No, conditional requires measurements on both.
        print("Completed conditional probability tests.")
    else:
        print("Not enough qubits for conditional probability test.")

def test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_3qubits_exp1.safetensors",
                          output_file = './assets/marginal_probability_heatmap_3qubits01.png',
                          device='cuda'):
    print("\n=== Test 2: Heatmap Marginal ===")
    backend_type = 'pytorch'
    backend = backend_factory.BackendFactory.create_backend(backend_type, device=device)
    engine = EngineSiamese(backend=backend)
    
    if not os.path.exists(qctn_cores_file):
        print(f"Warning: {qctn_cores_file} not found. Using random initialization.")
        qctn_graph = QCTNHelper.generate_example_graph()
        qctn = QCTN(qctn_graph, backend=engine.backend)
    else:
        qctn_graph = QCTNHelper.generate_example_graph()
        qctn = QCTN.from_pretrained(qctn_graph, qctn_cores_file, backend=engine.backend)
        
    edge_size = 100
    N = 1
    B = edge_size * edge_size
    D = qctn.nqubits
    K = 3

    # data_list = generate_Mx_phi_x_uniform(num_batch=N, batch_size=B, num_qubits=D, K=K, edge_size=edge_size, device=device)
    # measure_input_list = data_list[0][0]
    
    x = torch.empty((B, D), device=device)
    delta = 5 / edge_size
    step = 10 / edge_size
    for dx in range(edge_size):
        for dy in range(edge_size):
            vals = [dx * step - 5 + delta / 2, dy * step - 5 + delta / 2]
            if D > 2:
                vals += [0.0] * (D - 2)
            x[dx * edge_size + dy, :] = torch.tensor(vals, device=device)
    
    Mx_list, out = engine.generate_data(x, K=K)
    measure_input_list = Mx_list
    
    
    # Generate Uniform Data
    # data_list = generate_Mx_phi_x_uniform(num_batch=N, batch_size=B, num_qubits=D, K=K, edge_size=edge_size, device=device)
    # measure_input_list = data_list[0][0]
    
    circuit_states = generate_circuit_states_list(D, K, device=device)

    circuit_states = [engine.backend.convert_to_tensor(s) for s in circuit_states]
    measure_input_list = [engine.backend.convert_to_tensor(m) for m in measure_input_list]
    
    # Calculate Marginal Probability for first 2 qubits
    print("Calculating marginal probability for qubits [0, 1]...")
    result = engine.calculate_marginal_probability(
        qctn,
        circuit_states,
        [measure_input_list[0], measure_input_list[1]],
        [0, 1]
    )
    
    print(f"Result shape: {result.shape}")
    
    # Plot Heatmap
    heatmap = result.reshape(edge_size, edge_size).cpu().numpy()
    plt.figure()
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Marginal Probability Heatmap (q0, q1)')
    
    plt.savefig(output_file)
    print(f"Heatmap saved to {output_file}")

def test_sampling(qctn_cores_file="./assets/qctn_cores_3qubits_exp1.safetensors", output_file="./assets/samples_scatter.png", device='cuda'):
    print("\n=== Test 3: Sampling (Continuous) ===")
    backend_type = 'pytorch'
    backend = backend_factory.BackendFactory.create_backend(backend_type, device=device)
    engine = EngineSiamese(backend=backend)
    
    # Load QCTN
    if not os.path.exists(qctn_cores_file):
        print(f"Warning: {qctn_cores_file} not found. Using random initialization.")
        qctn_graph = QCTNHelper.generate_example_graph()
        qctn = QCTN(qctn_graph, backend=engine.backend)
    else:
        qctn_graph = QCTNHelper.generate_example_graph()
        qctn = QCTN.from_pretrained(qctn_graph, qctn_cores_file, backend=engine.backend)
    
    D = qctn.nqubits
    K = 3 # Dimension
    
    # Circuit states: |0> (or whatever initial state)
    circuit_states = generate_circuit_states_list(D, K, device=device)
    circuit_states = [engine.backend.convert_to_tensor(s) for s in circuit_states]
    
    num_samples = 1000  # Increased for better visualization
    
    print(f"Sampling {num_samples} samples from {D}-qubit circuit using Numerical Inverse CDF...")
    # Using bounds [-5, 5] and grid_size 1000 as per instructions
    t0 = time.time()
    samples = engine.sample(qctn, circuit_states, num_samples, K, bounds=[-5, 5], grid_size=100)
    t1 = time.time()
    
    print(f"Sampling finished in {t1-t0:.4f}s")
    print(f"Samples shape: {samples.shape}")
    print(f"First 5 samples:\n{samples[:5]}")
    
    # Plotting
    if samples.shape[1] >= 2 and output_file:
        # Move to CPU for plotting
        samples_np = samples.to('cpu').detach().numpy()
        x_vals = samples_np[:, 0]
        y_vals = samples_np[:, 1]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(x_vals, y_vals, alpha=0.6, s=10, c='blue', edgecolors='none')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.title(f"Sampled Points (N={num_samples})\nFirst 2 Qubits")
        plt.xlabel("Qubit 0")
        plt.ylabel("Qubit 1")
        plt.axhline(0, color='grey', linewidth=0.5)
        plt.axvline(0, color='grey', linewidth=0.5)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to {output_file}")
    
    # Basic checks
    assert samples.shape == (num_samples, D)
    # Check if samples are within bounds
    assert samples.min() >= -5.0
    assert samples.max() <= 5.0
    
    # Optional: Basic moment check
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
    print(f"Sample Mean per qubit: {mean}")
    print(f"Sample Std per qubit: {std}")
    
    print("Sampling test passed!")


if __name__ == "__main__":
    # test_random_probabilities(device='cpu')
    # test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_3qubits_exp2.safetensors",
    #                       output_file = './assets/marginal_probability_heatmap_3qubits_exp2_01.png',
    #                       device='cpu')
    # test_sampling(qctn_cores_file="./assets/qctn_cores_3qubits_exp2.safetensors", device='cpu')
    
    
    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores.safetensors",
                          output_file = './assets/marginal_probability_heatmap_3qubits_exp2_01.png',
                          device='cpu')
    test_sampling(qctn_cores_file="./assets/qctn_cores.safetensors", device='cpu')