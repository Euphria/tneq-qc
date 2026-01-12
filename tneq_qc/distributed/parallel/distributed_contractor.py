import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from copy import deepcopy
import opt_einsum
from ...core.tn_tensor import TNTensor

class DistributedHierarchicalContractor:
    """
    Contractor for distributed model parallel training with hierarchical reduction.
    
    Implements:
    1. Local subgraph contraction (Symmetric L-M-R structure)
    2. Hierarchical tree reduction for merging results
    """
    
    def __init__(self, engine, mpi, partition):
        self.engine = engine
        self.backend = engine.backend
        self.mpi = mpi
        self.partition = partition
        self.rank = mpi.rank
        self.size = mpi.world_size
        
        # Cache for global plan
        self._cached_plan = None
        self._cached_shapes = None

    def forward(self, qctn, circuit_states_list, measure_input_list):
        """
        Compute loss using distributed hierarchical contraction.
        
        Args:
            qctn: Quantum Circuit Tensor Network
            circuit_states_list: List of circuit states
            measure_input_list: List of measurement matrices
            
        Returns:
            loss: Scalar loss value
            local_grad_bundle: Data needed for backward pass (or computed gradients if manual)
        """
        # 1. Build or Retrieve Global Plan (Indices)
        plan = self._get_or_build_plan(qctn, circuit_states_list, measure_input_list)
        
        # 2. Prepare Local Tensors & Indices
        local_operands, local_indices = self._prepare_local_contraction(
            plan, qctn, circuit_states_list, measure_input_list
        )
        
        # 3. Local Contraction
        # Result has open indices corresponding to "Cut" edges
        local_matrix, current_indices = self._contract_local(local_operands, local_indices)
        
        # 4. Distributed Hierarchical Reduction
        # This computes the global scalar result (Trace)
        # We need to perform this reduction such that we can backpropagate.
        # Since we cannot backprop through MPI, we must implement the backward pass manually
        # OR assume the user handles it. 
        # Given "forward_with_gradient" in ModelParallelTrainer, we likely need to return gradients.
        # Here we return the loss and the local_matrix for backward steps.
        
        loss = self._distributed_reduce_forward(local_matrix, current_indices)
        
        return loss, local_matrix, current_indices

    def forward_with_gradient(self, qctn, circuit_states_list, measure_input_list):
        """
        Forward and Backward pass to compute gradients.
        
        Returns:
            loss: float
            local_grads: Dict[core_name, tensor]
        """
        # 1. Forward
        # Enable gradient tracking for local contraction
        with torch.set_grad_enabled(True):
            plan = self._get_or_build_plan(qctn, circuit_states_list, measure_input_list)
            local_operands, local_indices = self._prepare_local_contraction(
                plan, qctn, circuit_states_list, measure_input_list
            )
            
            # Record local cores involved for extraction later
            # (We only need minimal info since we use Autograd on the local block)
            # Actually, `local_operands` contains the core tensors. 
            # We need to retain references to them to extract .grad
            
            local_block, block_indices = self._contract_local(local_operands, local_indices)
            
        # 2. Distributed Forward Reduction (Manual, no Autograd)
        # Detach local_block for reduction, but keep it for backward
        local_block_detached = local_block.detach()
        if hasattr(local_block_detached, 'contiguous'):
             local_block_detached = local_block_detached.contiguous()

        # Perform reduction and store intermediate results for backward
        loss_val, reduction_ctx = self._distributed_reduce_forward_save_ctx(local_block_detached, block_indices)
        
        # 3. Distributed Backward (Manual)
        # Compute gradient w.r.t local_block
        grad_wrt_block = self._distributed_reduce_backward(reduction_ctx)
        
        # 4. Local Backward
        # Connect the manual gradient to the Autograd graph
        local_block.backward(grad_wrt_block)
        
        # 5. Extract gradients for local cores
        local_grads = {}
        for core_name in self.partition.local_core_names:
            weights = qctn.cores_weights[core_name]
            if isinstance(weights, TNTensor):
                if weights.tensor.grad is not None:
                    local_grads[core_name] = weights.tensor.grad
            else:
                 if weights.grad is not None:
                    local_grads[core_name] = weights.grad
                    
        return loss_val, local_grads

    def _get_or_build_plan(self, qctn, states, measures):
        # Identify global indices for every edge to ensure consistency across ranks
        # Simplified: Use deterministic hashing or consistent enumeration
        
        # We need to replicate the symmetry expansion (L, M, R)
        if self._cached_plan:
            return self._cached_plan
            
        plan = {'nodes': [], 'edges': {}}
        
        # Helper for unique edge ID
        # Format: "C:A:0-C:B:1" (Core A idx 0 connect to Core B idx 1)
        # Or simpler: we assign a unique char to every "Logical Bond" in the symmetric graph.
        
        # 1. Core-Core bonds (Left)
        # 2. Core-Core bonds (Right)
        # 3. Core-Mx-Core bonds (Vertical)
        
        # Enumerate all logical edges in QCTN
        # We assign a unique symbol to each edge in the adjacency list
        
        # Global Symbol Map
        # bond_key -> symbol
        symbol_map = {}
        counter = 0
        def get_sym():
            nonlocal counter
            s = opt_einsum.get_symbol(counter)
            counter += 1
            return s
            
        # QCTN edges (Adj Table)
        # Each entry in adjacency_table is a Core.
        # in_edge_list, out_edge_list contain connections.
        
        # We need to handle core_tensor_list construction exactly like GreedyStrategy
        # BUT assign global symbols.
        
        # Better strategy: 
        # Since we simply need consistent symbols, we can sort all cores/edges globally on every rank.
        # Since qctn is same on all ranks, this is deterministic.
        
        # Strategy:
        # 1. Iterate over all cores (sorted).
        # 2. For each core, iterate over edges.
        # 3. Assign symbol to connection. Connection defined by sorted( (src_core, src_idx), (dst_core, dst_idx) ).
        # For L side: symbols SL_...
        # For R side: symbols SR_...
        # For Vertical: symbols V_...
        
        # Let's build the plan
        bond_symbols = {} # (c1_name, c2_name, rank_idx) -> symbol. Sort c1, c2.
        
        # Pre-pass: identify all Core-Core connections
        for c1 in sorted(qctn.cores):
            # Access QCTN internal structure to find neighbors
            # adjacency_table has list of dicts.
            # We assume qctn structure allows finding neighbor name and edge indices
            # Let's reuse GreedyStrategy logic but JUST for the list construction step, 
            # and inject deterministic symbols.
            pass
            
        # Due to complexity, we will implement a simplified index generator
        # Assume 1D chain for now? No, need to be general or reuse general logic.
        
        self._cached_plan = self._build_greedy_compatible_plan(qctn, states, measures)
        return self._cached_plan

    def _build_greedy_compatible_plan(self, qctn, states, measures):
        # Re-implementation of GreedyStrategy graph building with Global Indexing
        core_tensor_list = []
        nqubits = qctn.nqubits
        
        # Global Symbol Counters 
        global_symbol_map = {} # key -> symbol
        sym_counter = 0
        def get_sym(key):
            nonlocal sym_counter
            if key not in global_symbol_map:
                global_symbol_map[key] = opt_einsum.get_symbol(sym_counter)
                sym_counter += 1
            return global_symbol_map[key]

        operands = [] 
        
        # Store connections for Mx which are discovered during Core traversal
        # qubit -> list of symbols
        mx_connections_l = {}
        mx_connections_r = {}

        # Iterate Cores
        for core_name in sorted(qctn.cores):
            weights = qctn.cores_weights[core_name]
            core_info = next(c for c in qctn.adjacency_table if c['core_name'] == core_name)
            
            l_indices = []
            r_indices = []
            
            # Process edges in order: In then Out (Standard TNTensor layout)
            all_edges = core_info['in_edge_list'] + core_info['out_edge_list']
            
            # Track which edges are inputs vs outputs for logic differentiation
            num_in = len(core_info['in_edge_list'])
            
            for i, edge in enumerate(all_edges):
                is_input = (i < num_in)
                
                if edge['neighbor_idx'] != -1:
                    # Core-Core
                    neighbor = edge['neighbor_name']
                    n1, n2 = sorted((core_name, neighbor))
                    bond_key = ("B", n1, n2, 0)
                    
                    l_indices.append(get_sym(("L", bond_key)))
                    r_indices.append(get_sym(("R", bond_key)))
                else:
                    # Open leg -> State (if In) or Mx (if Out)
                    q = edge['qubit_idx']
                    leg_idx = 0 # Simplified
                    
                    if is_input:
                        # Input Leg -> From State
                        state_key = ("State", core_name, q, leg_idx)
                        sl = get_sym(("L", state_key))
                        sr = get_sym(("R", state_key))
                        
                        l_indices.append(sl)
                        r_indices.append(sr)
                        
                        # Create State Operands immediately
                        if q < len(states) and states[q] is not None:
                             # Left State (Vector)
                             operands.append({
                                 'name': f"state_L_{q}",
                                 'type': 'state',
                                 'indices': sl, # State vector index
                                 'tensor_obj': states[q],
                                 'associated_core': core_name
                             })
                             # Right State (Vector)
                             operands.append({
                                 'name': f"state_R_{q}",
                                 'type': 'state',
                                 'indices': sr,
                                 'tensor_obj': states[q],
                                 'associated_core': core_name
                             })
                             
                    else:
                        # Output Leg -> To Mx
                        mx_key = ("Mx", core_name, q, leg_idx)
                        sl = get_sym(("L", mx_key))
                        sr = get_sym(("R", mx_key))
                        
                        l_indices.append(sl)
                        r_indices.append(sr)
                        
                        # Register connection for Mx creation later
                        if q not in mx_connections_l: mx_connections_l[q] = []
                        if q not in mx_connections_r: mx_connections_r[q] = []
                        mx_connections_l[q].append(sl)
                        mx_connections_r[q].append(sr)

            # Left Core
            operands.append({
                'name': core_name,
                'type': 'core',
                'side': 'L',
                'indices': "".join(l_indices),
                'tensor_obj': weights
            })
            
            # Right Core
            operands.append({
                'name': core_name,
                'type': 'core',
                'side': 'R',
                'indices': "".join(r_indices),
                'tensor_obj': weights
            })
            
        # Add Mx
        for q in range(qctn.nqubits):
            if q < len(measures) and measures[q] is not None:
                if q in mx_connections_l:
                    inds_l = "".join(mx_connections_l[q])
                    inds_r = "".join(mx_connections_r[q])
                    
                    indices = inds_l + inds_r
                    
                    # Handle batch
                    mx = measures[q]
                    if mx.ndim > 2: # Has batch
                         b_sym = get_sym("BATCH") # Global batch char
                         indices = b_sym + indices
                    
                    # Find associated core (any)
                    # We just use the first core that registered this q
                    assoc_core = None
                    # We need to find which core 'produced' this mx connection
                    # We can't easily track back from dict. But filtering uses 'associated_core'.
                    # Let's attach partition check logic: 
                    # If Mx connects to Core X, and Core X is local, include Mx.
                    # We have list of symbols. 
                    
                    # To support filtering, we need to know owners.
                    # Simplified: We treat Mx as local if ANY connected core is local?
                    # Or strictly if the qubit is owned?
                    # We used associated_core logic before.
                    # We need to store associate core in mx_connections?
                    # But it's fine, we can assume Mx is present if needed by local contraction. 
                    # Actually `_prepare_local_contraction` filters.
                    # We can attach a dummy associated core or list.
                    
                    # Let's find one core for simplicity
                    # In this graph, 1 core per qubit usually.
                    
                    # Search
                    found_core = None
                    for c in sorted(qctn.cores):
                        c_info = next(x for x in qctn.adjacency_table if x['core_name'] == c)
                        for e in c_info['out_edge_list']:
                             if e['neighbor_idx'] == -1 and e['qubit_idx'] == q:
                                 found_core = c
                                 break
                        if found_core: break
                    
                    operands.append({
                        'name': f"mx_{q}",
                        'type': 'mx',
                        'indices': indices,
                        'tensor_obj': mx,
                        'associated_core': found_core
                    })

        return operands

    def _prepare_local_contraction(self, plan, qctn, states, measures):
        local_ops = []
        local_input_indices = []
        
        # Select operands
        for op in plan:
            keep = False
            if op['type'] == 'core':
                if self.partition.is_local_core(op['name']):
                    keep = True
            elif op['type'] in ['mx', 'state']:
                 # Keep if associated core is local
                 if 'associated_core' in op and self.partition.is_local_core(op['associated_core']):
                     keep = True
            
            if keep:
                # Resolve object
                tensor = op['tensor_obj']
                if isinstance(tensor, TNTensor):
                    tensor = tensor.tensor # Extract raw for contraction
                    
                local_ops.append(tensor)
                local_input_indices.append(op['indices'])
        
        return local_ops, local_input_indices

    def _contract_local(self, operands, indices_list):
        # Use opt_einsum to find path and contract
        # We need output indices: All indices that appear exactly once in the *Local* set.
        # Actually, double check: A "Cut" bond appears once locally (at the cut).
        # Internal bonds appear twice (connect 2 local cores).
        # Open legs (physical) connect to Mx (which is local), so they appear twice.
        # So yes, indices appearing once are the Cut Interface.
        
        if not operands:
             raise ValueError("No local operands found!")

        # Count indices
        counts = {}
        for s in indices_list:
            for c in s:
                counts[c] = counts.get(c, 0) + 1
        
        output_indices = [c for c, n in counts.items() if n == 1]
        output_indices.sort() # Deterministic order
        
        out_str = "".join(output_indices)
        
        # Call opt_einsum
        # inputs: op1, ind1, op2, ind2, ..., out_ind
        args = []
        for op, ind in zip(operands, indices_list):
            args.append(op)
            args.append(ind)
        args.append(out_str)
        
        result = opt_einsum.contract(*args)
        
        return result, out_str

    def _distributed_reduce_forward(self, local_matrix, current_indices):
        # Implement call to _distributed_reduce_forward_save_ctx but ignore ctx
        loss, _ = self._distributed_reduce_forward_save_ctx(local_matrix, current_indices)
        return loss
        
    def _distributed_reduce_forward_save_ctx(self, local_matrix, current_indices):
        # Tree reduction.
        # Rank 0 is root.
        
        # For simplicity in this example, and since user asked for "Adjacent 2 contract",
        # We can use a doubling communication pattern.
        # Step 1: 0-1, 2-3, ... (Partner: rank ^ 1)
        # Step 2: 0-2, 4-6, ... (Partner: rank ^ 2)
        # ...
        
        # We save inputs at each step to context for backward.
        ctx = {'steps': [], 'rank': self.rank, 'initial': local_matrix, 'indices': current_indices}
        
        current_data = local_matrix
        current_ind = current_indices
        
        num_steps = self.size.bit_length() # Assuming power of 2
        if self.size & (self.size-1) != 0:
             # Handle non-power-of-2 if needed, but example uses 4 or 16.
             pass
             
        for i in range(num_steps):
            mask = 1 << i
            # Check if active
            # Active if rank % (2 * mask) == 0  -> Receiver
            # Or rank % (2 * mask) == mask -> Sender
            
            group_base = self.rank - (self.rank % (2 * mask))
            partner = self.rank ^ mask
            
            if partner >= self.size: continue
            
            step_info = {'step': i, 'role': None, 'partner': partner}
            
            if (self.rank & mask) == 0:
                # I am Receiver (Low Rank)
                # Receive from Partner (High Rank)
                other_data_np = self.mpi.comm.recv(source=partner, tag=i)
                other_ind = self.mpi.comm.recv(source=partner, tag=i+100)
                
                # Convert to tensor on correct device
                # Use engine's backend mechanism
                other_data = self.backend.convert_to_tensor(other_data_np)
                
                # Perform Contraction (Tensor Parallel style?)
                # "Adjacent 2 contract"
                # Contraction of current_data and other_data.
                # Auto detect indices
                
                # Update current_data
                # We save Left and Right inputs for backward
                # Left is Me, Right is Partner (since I am Low Rank)
                # Wait, structure is linear 0, 1, 2, 3.
                # So indices of 0 and 1 should share a bond.
                
                new_data, new_ind = self._contract_pair(current_data, current_ind, other_data, other_ind)
                
                step_info['role'] = 'recv'
                step_info['my_data'] = current_data # Save pre-contraction state
                step_info['other_data'] = other_data
                step_info['other_ind'] = other_ind
                step_info['my_ind'] = current_ind
                
                current_data = new_data
                current_ind = new_ind
                
            else:
                # I am Sender (High Rank)
                # Send to Partner (Low Rank)
                # Ensure data is on CPU/Contiguous before send
                data_np = current_data.detach().cpu().numpy()
                self.mpi.comm.send(data_np, dest=partner, tag=i)
                self.mpi.comm.send(current_ind, dest=partner, tag=i+100)
                
                step_info['role'] = 'send'
                # Sender is done in reduction, but waits for gradient in backward
                current_data = None # Clear to save memory?
            
            ctx['steps'].append(step_info)
            
            if step_info['role'] == 'send':
                break
                
        return current_data, ctx

    def _distributed_reduce_backward(self, ctx):
        # Replays steps in reverse
        # Final result gradient is usually 1.0 (for Loss)
        
        device = getattr(self.backend, 'device', 'cpu')
        # Fallback if backend wrapper doesn't expose device directly (e.g. wrapper around torch)
        if device == 'cpu' and hasattr(self.backend, 'backend_info'):
             device = self.backend.backend_info.device

        current_grad = torch.tensor(1.0, device=device) if ctx.get('current_data') is not None else torch.tensor(1.0, device=device)
        # Wait, ctx['current_data'] is the FINAL result on Rank 0.
        # Other ranks have current_data=None after send.
        
        # Rank 0 has the final scalar.
        if self.rank == 0:
             current_grad = torch.tensor(1.0, device=device)
        else:
             current_grad = None
             
        steps = ctx['steps'][::-1] # Reverse
        
        for step_info in steps:
            partner = step_info['partner']
            i = step_info['step']
            
            if step_info['role'] == 'recv':
                # I computed C = Contract(A, B). I have grad dC.
                # I need dA and dB.
                # A = my_data, B = other_data
                
                A = step_info['my_data']
                B = torch.tensor(step_info['other_data'], device=device) # It was numpy
                
                # Re-run contraction to get grads
                # We use Autograd locally for this step!
                with torch.enable_grad():
                    A.requires_grad_(True)
                    B.requires_grad_(True)
                    C, _ = self._contract_pair(A, step_info['my_ind'], B, step_info['other_ind'])
                    C.backward(current_grad)
                    
                    dA = A.grad
                    dB = B.grad
                
                # I keep dA (my path). Send dB to partner.
                current_grad = dA
                
                # Send dB
                dB_np = dB.detach().cpu().numpy()
                self.mpi.comm.send(dB_np, dest=partner, tag=i+200)
                
            elif step_info['role'] == 'send':
                # I sent my data. Now I receive gradient.
                grad_np = self.mpi.comm.recv(source=partner, tag=i+200)
                current_grad = torch.from_numpy(grad_np).to(device)
            
        return current_grad
        
    def _contract_pair(self, A, indA, B, indB):
        # find common indices
        common = set(indA) & set(indB)
        # output = (indA + indB) - common (unique only) ?
        # Standard matrix mult rules.
        # Actually opt_einsum handles this.
        args = [A, indA, B, indB]
        path_info = opt_einsum.contract_path(*args) # Check path
        out_ind = path_info[1].output_subscript
        res = opt_einsum.contract(*args)
        return res, out_ind

