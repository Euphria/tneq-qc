"""
TNGraph - Tensor Network Graph representation and manipulation.

This module provides a class to represent quantum tensor networks as ASCII art
and convert between string representation and structured data.
"""

import random
import string
from typing import List, Tuple, Optional
import re


class TNGraph:
    """
    Tensor Network Graph representation.
    
    Represents a quantum circuit tensor network where:
    - Each row represents a qubit line
    - Each letter (A-Z) represents a tensor
    - Numbers represent bond dimensions between tensors or at edges
    
    Example:
        -2-------B--5--C--3--D-------2-
        -2-A-4---------------D-------2-
        -2-A--4--B--7--C--2--D--4--E-2-
        -2-A--3--B--6--------------E-2-
        -2-------------C--8--------E-2-
    
    Internal representation:
        Each qubit line is a list of tuples: (tensor_name, left_bond, right_bond)
        Example: [('B', 2, 5), ('C', 5, 3), ('D', 3, 2)]
    """
    
    def __init__(self, graph_string: Optional[str] = None, n_qubits: int = 0):
        """
        Initialize TNGraph.
        
        Args:
            graph_string: Optional ASCII art string representation
            n_qubits: Number of qubit lines (if creating empty graph)
        """
        self.n_qubits = n_qubits
        self.n_tensors = 0
        self.tensor_names = []  # List of tensor names in order
        
        # Main data structure: list of lists
        # graph[i] = [(tensor_name, left_bond, right_bond), ...]
        self.graph: List[List[Tuple[str, int, int]]] = []
        
        if graph_string:
            self.from_string(graph_string)
        else:
            self.graph = [[] for _ in range(n_qubits)]
    
    def from_string(self, graph_string: str) -> 'TNGraph':
        """
        Parse ASCII art string and populate the graph structure.
        
        Args:
            graph_string: Multi-line string representation of the tensor network
            
        Returns:
            self: For method chaining
            
        Example:
            >>> graph = TNGraph()
            >>> graph.from_string('''
            ... -2-------B--5--C--3--D-------2-
            ... -2-A-4---------------D-------2-
            ... ''')
        """
        lines = [line.strip() for line in graph_string.strip().split('\n') if line.strip()]
        self.n_qubits = len(lines)
        self.graph = [[] for _ in range(self.n_qubits)]
        
        tensor_set = set()
        
        for qubit_idx, line in enumerate(lines):
            # Remove leading/trailing dashes and edge bonds
            line = line.strip('-')
            
            # Pattern to match: number-letter-number or just letter with bonds
            # Examples: "2-A-4", "B--5", "5--C--3"
            
            # Split by dashes and process segments
            segments = self._parse_line(line)
            
            for tensor_name, left_bond, right_bond in segments:
                self.graph[qubit_idx].append((tensor_name, left_bond, right_bond))
                tensor_set.add(tensor_name)
        
        # Sort and store tensor names
        self.tensor_names = sorted(list(tensor_set))
        self.n_tensors = len(self.tensor_names)
        
        return self
    
    def _parse_line(self, line: str) -> List[Tuple[str, int, int]]:
        """
        Parse a single qubit line and extract tensors with their bonds.
        
        Handles cases like A----B (no connection) by setting bonds to 0.
        
        Args:
            line: A single line string (without leading/trailing edge bonds)
            
        Returns:
            List of (tensor_name, left_bond, right_bond) tuples
        """
        result = []
        
        # Find all tensors (uppercase letters) with their positions
        tensor_positions = []
        for i, char in enumerate(line):
            if char.isupper():
                tensor_positions.append((i, char))
        
        if not tensor_positions:
            return result
        
        for idx, (pos, tensor_name) in enumerate(tensor_positions):
            # Determine left bond
            if idx == 0:
                # First tensor - look left for edge bond
                left_part = line[:pos]
                left_bond = self._extract_bond_from_string(left_part, 'right')
            else:
                # Look between previous tensor and this tensor
                prev_pos = tensor_positions[idx-1][0]
                between = line[prev_pos+1:pos]
                left_bond = self._extract_bond_from_string(between, 'right')
            
            # Determine right bond
            if idx == len(tensor_positions) - 1:
                # Last tensor - look right for edge bond
                right_part = line[pos+1:]
                right_bond = self._extract_bond_from_string(right_part, 'left')
            else:
                # Look between this tensor and next tensor
                next_pos = tensor_positions[idx+1][0]
                between = line[pos+1:next_pos]
                right_bond = self._extract_bond_from_string(between, 'left')
            
            result.append((tensor_name, left_bond, right_bond))
        
        return result
    
    def _extract_bond_from_string(self, s: str, side: str) -> int:
        """
        Extract bond dimension from a string segment.
        
        If no number found (e.g., "----"), returns 0 (no connection).
        
        Args:
            s: String segment between tensors or at edges
            side: 'left' or 'right' - which side to prefer if multiple numbers
            
        Returns:
            Bond dimension (0 if no connection)
        """
        # Remove dashes and find numbers
        numbers = re.findall(r'\d+', s)
        
        if not numbers:
            return 0  # No connection (e.g., A----B)
        
        # If multiple numbers, take the one from the specified side
        if side == 'left':
            return int(numbers[0])
        else:
            return int(numbers[-1])
    
    def to_string(self, spacing: int = 2) -> str:
        """
        Convert the graph structure to ASCII art string representation.
        
        All lines have the same length by leaving space for missing tensors.
        Two tensors are always separated by 5 dashes.
        - No connection: A-----B (5 dashes)
        - With connection: A--3--B (bond number centered in middle 3 positions)
        
        Args:
            spacing: Ignored (kept for compatibility), always uses 5-dash spacing
            
        Returns:
            Multi-line string representation of the tensor network
            
        Example:
            >>> print(graph.to_string())
            -2-------B--5--C--3--D-------2-
            -2-A-4---------------D-------2-
        """
        # First pass: determine which tensors appear on each line
        # Create a position map for all tensors across all lines
        tensor_positions = {}  # tensor_name -> list of (qubit_idx, position_in_line)
        
        for qubit_idx in range(self.n_qubits):
            for pos, (tensor_name, _, _) in enumerate(self.graph[qubit_idx]):
                if tensor_name not in tensor_positions:
                    tensor_positions[tensor_name] = []
                tensor_positions[tensor_name].append((qubit_idx, pos))
        
        # Determine the order of all tensors: from A to the maximum tensor name
        # For example, if max tensor is 'E', all_tensors_ordered = ['A', 'B', 'C', 'D', 'E']
        if tensor_positions:
            max_tensor = max(tensor_positions.keys())
            all_tensors_ordered = list(string.ascii_uppercase[:ord(max_tensor) - ord('A') + 1])
        else:
            all_tensors_ordered = []
        
        lines = []
        
        for qubit_idx in range(self.n_qubits):
            # Build a map of which tensors appear on this line
            line_tensors = {}  # tensor_name -> (left_bond, right_bond)
            for tensor_name, left_bond, right_bond in self.graph[qubit_idx]:
                line_tensors[tensor_name] = (left_bond, right_bond)
            
            line_parts = []
            
            # Start with left edge bond
            if self.graph[qubit_idx]:
                first_tensor = self.graph[qubit_idx][0]
                left_edge = first_tensor[1] if first_tensor[1] > 0 else 2
            else:
                left_edge = 2
            line_parts.append(f"-{left_edge}-")
            
            tot_exists = 0
            # Process each tensor position in global order
            for i, tensor_name in enumerate(all_tensors_ordered):

                if tensor_name in line_tensors:
                    tot_exists += 1

                    # This tensor appears on this line
                    left_bond, right_bond = line_tensors[tensor_name]

                    # Add connection to previous position
                    # if tot_exists > 1:
                    if i > 0:
                        # Check if previous tensor exists on this line
                        prev_tensor = all_tensors_ordered[i-1]
                        if prev_tensor in line_tensors:
                            # Both tensors exist - use right bond of previous
                            prev_right = line_tensors[prev_tensor][1]
                            bond_value = prev_right
                        else:
                            # Previous tensor doesn't exist on this line
                            # Use left bond of current tensor
                            bond_value = left_bond
                        
                        # Format the connection (5 dashes total)
                        if bond_value > 0 and tot_exists > 1:
                            bond_str = str(bond_value)
                            if len(bond_str) == 1:
                                line_parts.append(f"--{bond_str}--")
                            elif len(bond_str) == 2:
                                line_parts.append(f"-{bond_str}--")
                            else:  # 3 or more digits
                                line_parts.append(f"-{bond_str}-")
                        else:
                            line_parts.append("-----")
                    
                    # Add tensor name
                    line_parts.append(tensor_name)
                else:
                    # This tensor doesn't appear on this line - leave space
                    if i > 0:
                        line_parts.append("-----")  # Connection space
                    line_parts.append("-")  # Tensor space (just a dash)

            # End with right edge bond
            if self.graph[qubit_idx]:
                last_tensor = self.graph[qubit_idx][-1]
                right_edge = last_tensor[2] if last_tensor[2] > 0 else 2
            else:
                right_edge = 2
            line_parts.append(f"-{right_edge}-")
            
            lines.append(''.join(line_parts))
        
        return '\n'.join(lines)
    
    def set_from_string(self, graph_string: str) -> 'TNGraph':
        """
        Alias for from_string() for clarity.
        
        Args:
            graph_string: Multi-line string representation
            
        Returns:
            self: For method chaining
        """
        return self.from_string(graph_string)
    
    def export_to_string(self, spacing: int = 2) -> str:
        """
        Alias for to_string() for clarity.
        
        Args:
            spacing: Number of dashes between elements
            
        Returns:
            Multi-line string representation
        """
        return self.to_string(spacing)
    
    def get_tensor_qubits(self, tensor_name: str) -> List[int]:
        """
        Get all qubit indices that a tensor acts on.
        
        Args:
            tensor_name: Name of the tensor (e.g., 'A', 'B')
            
        Returns:
            List of qubit indices (0-indexed)
        """
        qubits = []
        for qubit_idx in range(self.n_qubits):
            for name, _, _ in self.graph[qubit_idx]:
                if name == tensor_name:
                    qubits.append(qubit_idx)
                    break
        return qubits
    
    def get_tensor_info(self, tensor_name: str) -> dict:
        """
        Get detailed information about a specific tensor.
        
        Args:
            tensor_name: Name of the tensor
            
        Returns:
            Dictionary with tensor information:
            - qubits: list of qubit indices
            - bonds: list of (qubit_idx, left_bond, right_bond)
        """
        qubits = []
        bonds = []
        
        for qubit_idx in range(self.n_qubits):
            for name, left, right in self.graph[qubit_idx]:
                if name == tensor_name:
                    qubits.append(qubit_idx)
                    bonds.append((qubit_idx, left, right))
        
        return {
            'qubits': qubits,
            'bonds': bonds
        }
    
    # def add_tensor(self, tensor_name: str, qubit_idx: int, 
    #                left_bond: int, right_bond: int, position: Optional[int] = None):
    #     """
    #     Add a tensor to a specific qubit line.
        
    #     Args:
    #         tensor_name: Name of the tensor (single uppercase letter)
    #         qubit_idx: Which qubit line (0-indexed)
    #         left_bond: Left bond dimension
    #         right_bond: Right bond dimension
    #         position: Where to insert in the line (None = append at end)
    #     """
    #     if qubit_idx >= self.n_qubits:
    #         raise ValueError(f"Qubit index {qubit_idx} out of range (n_qubits={self.n_qubits})")
        
    #     tensor_tuple = (tensor_name, left_bond, right_bond)
        
    #     if position is None:
    #         self.graph[qubit_idx].append(tensor_tuple)
    #     else:
    #         self.graph[qubit_idx].insert(position, tensor_tuple)
        
    #     if tensor_name not in self.tensor_names:
    #         self.tensor_names.append(tensor_name)
    #         self.tensor_names.sort()
    #         self.n_tensors = len(self.tensor_names)
    
    # def remove_tensor(self, tensor_name: str):
    #     """
    #     Remove all instances of a tensor from all qubit lines.
        
    #     Args:
    #         tensor_name: Name of the tensor to remove
    #     """
    #     for qubit_idx in range(self.n_qubits):
    #         self.graph[qubit_idx] = [
    #             t for t in self.graph[qubit_idx] if t[0] != tensor_name
    #         ]
        
    #     if tensor_name in self.tensor_names:
    #         self.tensor_names.remove(tensor_name)
    #         self.n_tensors = len(self.tensor_names)
    
    def modify_bond(self, qubit_idx: int, tensor_name: str, new_value: int):
        """
        Modify the right bond value of a specific tensor on a qubit line.
        The tensor cannot be the last one on the line.
        
        Args:
            qubit_idx: Which qubit line (0-indexed)
            tensor_name: Name of the tensor to modify
            new_value: New bond dimension value for the right link
            
        Example:
            # Change A--3--C to A--4--C
            graph.modify_bond(0, 'A', 4)
            
            # Change A--3--C to A--0--C (no connection)
            graph.modify_bond(0, 'A', 0)
            
        Raises:
            ValueError: If tensor is not found, or is the last tensor on the line
        """
        if qubit_idx >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit_idx} out of range")
        
        # Find the tensor
        tensor_idx = None
        for i, (name, left, right) in enumerate(self.graph[qubit_idx]):
            if name == tensor_name:
                tensor_idx = i
                break
        
        if tensor_idx is None:
            raise ValueError(f"Tensor {tensor_name} not found on qubit {qubit_idx}")
        
        # Check if it's the last tensor
        if tensor_idx == len(self.graph[qubit_idx]) - 1:
            raise ValueError(f"Cannot modify bond of {tensor_name}: it's the last tensor on qubit {qubit_idx}")
        
        # Modify the right bond
        name, left, right = self.graph[qubit_idx][tensor_idx]
        self.graph[qubit_idx][tensor_idx] = (name, left, new_value)
        
        # Also update the left bond of the next tensor to maintain consistency
        next_name, next_left, next_right = self.graph[qubit_idx][tensor_idx + 1]
        self.graph[qubit_idx][tensor_idx + 1] = (next_name, new_value, next_right)
    
    def remove_tensor_from_qubit(self, qubit_idx: int, tensor_name: str, bond_mode: str = 'min'):
        """
        Remove a tensor from a specific qubit line and reconnect neighbors.
        
        When removing a tensor between two others, uses the smaller bond value.
        If at edge, uses edge bond value of 2.
        
        Args:
            qubit_idx: Which qubit line (0-indexed)
            tensor_name: Name of the tensor to remove
            bond_mode: 'min' (default) - ['min', 'max', 'left', 'right']
            
        Examples:
            A--3--B--4--C  →  A--3--C  (remove B, use min(3,4)=3)
            -2-A--3--B-2-  →  -2-A-2-  (remove B, use edge bond 2)
        """
        if qubit_idx >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit_idx} out of range")
        
        # Find the tensor position
        tensor_idx = None
        for i, (name, _, _) in enumerate(self.graph[qubit_idx]):
            if name == tensor_name:
                tensor_idx = i
                break
        
        if tensor_idx is None:
            raise ValueError(f"Tensor {tensor_name} not found on qubit {qubit_idx}")
        
        # Get the tensor info
        _, left_bond, right_bond = self.graph[qubit_idx][tensor_idx]
        
        # Determine the new bond value for reconnection
        if tensor_idx == 0 and len(self.graph[qubit_idx]) > 1:
            # First tensor - reconnect next tensor to left edge
            # Use edge bond (left_bond or default 2)
            # eg. -2-A--3--B-2-  →  -2-B-2-
            new_bond = left_bond if left_bond > 0 else 2
            next_tensor = self.graph[qubit_idx][1]
            self.graph[qubit_idx][1] = (next_tensor[0], new_bond, next_tensor[2])
        
        elif tensor_idx == len(self.graph[qubit_idx]) - 1 and len(self.graph[qubit_idx]) > 1:
            # Last tensor - reconnect previous tensor to right edge
            # Use edge bond (right_bond or default 2)
            # eg. -2-A--3--B-2-  →  -2-A-2-
            new_bond = right_bond if right_bond > 0 else 2
            prev_tensor = self.graph[qubit_idx][tensor_idx - 1]
            self.graph[qubit_idx][tensor_idx - 1] = (prev_tensor[0], prev_tensor[1], new_bond)
        
        elif 0 < tensor_idx and tensor_idx < len(self.graph[qubit_idx]) - 1:
            # Middle tensor - reconnect neighbors
            # Use smaller of the two bonds, or 2 if both are 0
            # eg. A--3--B--4--C  →  A--3--C

            if bond_mode == 'min':
                new_bond = min(left_bond, right_bond)
            elif bond_mode == 'max':
                new_bond = max(left_bond, right_bond)
            elif bond_mode == 'left':
                new_bond = left_bond
            elif bond_mode == 'right':
                new_bond = right_bond
            else:
                raise ValueError(f"Invalid bond_mode '{bond_mode}': must be one of ['min', 'max', 'left', 'right']")

            prev_tensor = self.graph[qubit_idx][tensor_idx - 1]
            next_tensor = self.graph[qubit_idx][tensor_idx + 1]
            
            self.graph[qubit_idx][tensor_idx - 1] = (prev_tensor[0], prev_tensor[1], new_bond)
            self.graph[qubit_idx][tensor_idx + 1] = (next_tensor[0], new_bond, next_tensor[2])
        
        # Remove the tensor
        self.graph[qubit_idx].pop(tensor_idx)
        
        # TODO: n_tensors use max value?
        # Update tensor list if this was the last instance
        if not any(tensor_name in [t[0] for t in line] for line in self.graph):
            if tensor_name in self.tensor_names:
                self.tensor_names.remove(tensor_name)
                self.n_tensors = len(self.tensor_names)
    
    def insert_tensor_after(self, qubit_idx: int, tensor_name: str, insert_mode: str = 'random'):
        """
        Insert a new tensor to the right of the specified tensor on a qubit line.
        If tensor_name is "" (empty string), insert at the leftmost position.
        
        The new tensor name is automatically determined based on the current line:
        - Only considers tensors already on the current qubit line
        - Determines the insertion position first
        - Calculates available letters based on neighbors at that position
        
        Args:
            qubit_idx: Which qubit line (0-indexed)
            tensor_name: Name of the tensor to insert after, or "" to insert at start
            insert_mode: 'random' (default) - choose randomly from available names ['random', 'first', 'last', 'middle']
            
        Example:
            # Line: -2-A--3--C-2- (graph has 5 total tensors)
            # insert_tensor_after(0, 'A') → can insert B (between A and C)
            # insert_tensor_after(0, 'C') → can insert D or E (after C)
            # insert_tensor_after(0, '') → cannot insert (no letter before A)
            
        Raises:
            ValueError: If tensor not found, or no available tensor name exists
        """

        def choose_from_available(available: List[str], mode: str) -> str:
            if not available:
                raise ValueError("No available tensor names to choose from")
            if mode == 'random':
                return random.choice(available)
            elif mode == 'first':
                return available[0]
            elif mode == 'last':
                return available[-1]
            elif mode == 'middle':
                return available[len(available) // 2]
            else:
                raise ValueError(f"Invalid insert_mode '{mode}': must be one of ['random', 'first', 'last', 'middle']")

        if qubit_idx >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit_idx} out of range")
        
        # Get tensors on current line
        line_tensors = [name for name, _, _ in self.graph[qubit_idx]]
        
        # Determine maximum allowed tensor count based on graph total
        max_tensors = min(self.n_tensors + 1, 26)
        all_possible_letters = string.ascii_uppercase[:max_tensors]
        
        if tensor_name == "":
            # Insert at the leftmost position
            if not self.graph[qubit_idx]:
                # Empty line - use first available letter from all possible
                available = all_possible_letters
                if not available:
                    raise ValueError(f"No available tensor names on this line")

                # random choice
                new_tensor_name = choose_from_available(available, insert_mode)

                self.graph[qubit_idx].append((new_tensor_name, 2, 2))
            else:
                # Need to insert before the first tensor
                first_tensor = self.graph[qubit_idx][0]
                first_name = first_tensor[0]
                
                # Available letters: must be < first_name and not used on this line
                available_before = [l for l in all_possible_letters 
                                   if l < first_name and l not in line_tensors]
                
                if not available_before:
                    raise ValueError(
                        f"Cannot insert at start: no available tensor name before '{first_name}' "
                        f"within the allowed range (max: {max_tensors} tensors total)"
                    )
                
                # random choose available letter before first_name
                new_tensor_name = choose_from_available(available_before, insert_mode)
                
                # Get edge bond from first tensor
                edge_bond = first_tensor[1] if first_tensor[1] > 0 else 2
                
                # Insert new tensor at position 0
                new_tensor = (new_tensor_name, edge_bond, edge_bond)
                self.graph[qubit_idx].insert(0, new_tensor)
                
                # Update first tensor's left bond
                self.graph[qubit_idx][1] = (first_name, edge_bond, first_tensor[2])
        else:
            # Insert after the specified tensor
            # Find the tensor position
            tensor_idx = None
            for i, (name, _, _) in enumerate(self.graph[qubit_idx]):
                if name == tensor_name:
                    tensor_idx = i
                    break
            
            if tensor_idx is None:
                raise ValueError(f"Tensor {tensor_name} not found on qubit {qubit_idx}")
            
            # Check if it's the last tensor
            is_last = (tensor_idx == len(self.graph[qubit_idx]) - 1)
            
            current_tensor = self.graph[qubit_idx][tensor_idx]
            current_name, current_left, current_right = current_tensor
            
            if is_last:
                # Inserting after the last tensor
                # Available letters: must be > current_name and not used on this line
                available_after = [l for l in all_possible_letters 
                                  if l > current_name and l not in line_tensors]
                
                if not available_after:
                    raise ValueError(
                        f"Cannot insert after '{current_name}': no available tensor name after it "
                        f"within the allowed range (max: {max_tensors} tensors total)"
                    )
                
                # choose from available
                new_tensor_name = choose_from_available(available_after, insert_mode)
                
                # Get edge bond
                edge_bond = current_right if current_right > 0 else 2
                
                # Insert new tensor at the end
                new_tensor = (new_tensor_name, edge_bond, edge_bond)
                self.graph[qubit_idx].append(new_tensor)
                
                # Update current tensor's right bond
                self.graph[qubit_idx][tensor_idx] = (current_name, current_left, edge_bond)
            else:
                # Inserting between current and next tensor
                next_tensor = self.graph[qubit_idx][tensor_idx + 1]
                next_name, next_left, next_right = next_tensor
                
                # Available letters: must be between current_name and next_name, not used on this line
                available_between = [l for l in all_possible_letters 
                                    if current_name < l < next_name and l not in line_tensors]
                
                if not available_between:
                    raise ValueError(
                        f"Cannot insert between '{current_name}' and '{next_name}': "
                        f"no available tensor name between them "
                        f"within the allowed range (max: {max_tensors} tensors total)"
                    )
                
                # Use the smallest available letter in the range (closest to current)
                new_tensor_name = choose_from_available(available_between, insert_mode)
                
                # Get the bond value from the connection
                bond_value = current_right
                
                # Insert new tensor
                new_tensor = (new_tensor_name, bond_value, bond_value)
                self.graph[qubit_idx].insert(tensor_idx + 1, new_tensor)
                
                # No need to update current or next - they already have correct bonds
        
        # Update tensor list
        if new_tensor_name not in self.tensor_names:
            self.tensor_names.append(new_tensor_name)
            self.tensor_names.sort()
            self.n_tensors = len(self.tensor_names)
    
    def __str__(self) -> str:
        """String representation."""
        return self.to_string()
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"TNGraph(n_qubits={self.n_qubits}, n_tensors={self.n_tensors}, " \
               f"tensors={self.tensor_names})"
    
    def summary(self) -> str:
        """
        Get a summary of the tensor network.
        
        Returns:
            Human-readable summary string
        """
        summary_lines = [
            f"Tensor Network Graph",
            f"  Qubits: {self.n_qubits}",
            f"  Tensors: {self.n_tensors} ({', '.join(self.tensor_names)})",
            f"  Structure:"
        ]
        
        for qubit_idx in range(self.n_qubits):
            tensors = [t[0] for t in self.graph[qubit_idx]]
            summary_lines.append(f"    Qubit {qubit_idx}: {tensors if tensors else 'empty'}")
        
        return '\n'.join(summary_lines)

# Example usage and tests
if __name__ == "__main__":
    # Example 1: Parse from string
    print("=" * 60)
    print("Example 1: Parse from string")
    print("=" * 60)
    
    graph_str = """
-2-------B--5--C--3--D-------2-
-2-A-4---------------D-------2-
-2-A--4--B--7--C--2--D--4--E-2-
-2-A--3--B--6--------------E-2-
-2-------------C--8--------E-2-
    """
    
    graph = TNGraph()
    graph.from_string(graph_str)
    
    print(graph.summary())
    print()
    
    # Example 2: Access internal structure
    print("=" * 60)
    print("Example 2: Internal structure")
    print("=" * 60)
    
    for i in range(graph.n_qubits):
        print(f"Qubit {i}: {graph.graph[i]}")
    print()
    
    # Example 3: Get tensor information
    print("=" * 60)
    print("Example 3: Tensor information")
    print("=" * 60)
    
    for tensor_name in graph.tensor_names:
        info = graph.get_tensor_info(tensor_name)
        print(f"Tensor {tensor_name}:")
        print(f"  Acts on qubits: {info['qubits']}")
        print(f"  Bonds: {info['bonds']}")
    print()
    
    # Example 4: Export back to string
    print("=" * 60)
    print("Example 4: Export to string")
    print("=" * 60)
    
    exported = graph.export_to_string()
    print(exported)
    print()
    