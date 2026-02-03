using QuantumClifford
using QuantumClifford: logicalxview, logicalzview, MixedDestabilizer, Stabilizer, 
                       PauliOperator, nqubits, rank, tab, stab_to_gf2, comm,
                       stabilizerview, destabilizerview
using LinearAlgebra
using LinearAlgebra: I

#=============================================================================
    Helper Functions for Pauli Operator Support Detection
=============================================================================#

"""
    pauli_support(P::PauliOperator) -> Vector{Int}

Return the support of a Pauli operator, i.e., the set of qubit indices where P acts non-trivially (not identity).

# Example
```julia
julia> pauli_support(P"_XY_Z")
[2, 3, 5]
```
"""
function pauli_support(P::PauliOperator)
    n = nqubits(P)
    support = Int[]
    for i in 1:n
        x, z = P[i]
        if x || z  # Not identity if either X or Z bit is set
            push!(support, i)
        end
    end
    return support
end

"""
    is_supported_on(P::PauliOperator, A::Vector{Int}) -> Bool

Check if Pauli operator P is fully supported on subsystem A, 
i.e., P acts as identity on all qubits outside A.
"""
function is_supported_on(P::PauliOperator, A::Vector{Int})
    support = pauli_support(P)
    return all(q ∈ A for q in support)
end

#=============================================================================
    Logical Pauli Operators Generation
=============================================================================#

"""
    all_logical_paulis(stab::MixedDestabilizer) -> Vector{PauliOperator}

Generate all 4^k logical Pauli operators for a stabilizer code with k logical qubits.

Given a stabilizer code with k logical qubits, there are 4^k logical Pauli operators
corresponding to all combinations of I, X, Y, Z on each logical qubit. These operators
form a basis for the logical operator algebra M.

**Important**: Logical operators are defined as equivalence classes modulo the stabilizer
group. That is, if L is a logical operator and S is any stabilizer, then L*S is an 
equivalent logical operator. This function returns one representative from each 
equivalence class (the "bare" logical operators from logicalxview and logicalzview).

For finding logical operators supported on a subregion, use `subregion_logical_paulis`
which searches for representatives in each equivalence class that have the desired support.

The encoding is:
- 0 → I (identity)
- 1 → X (logical X)  
- 2 → Y (logical Y = iXZ)
- 3 → Z (logical Z)

# Arguments
- `stab::MixedDestabilizer`: A MixedDestabilizer representing the stabilizer code

# Returns
- `Vector{PauliOperator}`: All 4^k logical Pauli operators (one representative per class)

# Example
```julia
julia> s = S"XXXX
             ZIZI
             IZIZ"  # [[4,1,2]] code
julia> stab = MixedDestabilizer(s)
julia> ops = all_logical_paulis(stab)
julia> length(ops)  # 4^1 = 4
4
```
"""
function all_logical_paulis(stab::MixedDestabilizer)
    r = rank(stab)
    n = nqubits(stab)
    k = n - r  # number of logical qubits
    
    if k == 0
        # No logical qubits, return only identity
        return [zero(PauliOperator, n)]
    end
    
    LX = logicalxview(stab)
    LZ = logicalzview(stab)
    
    ops = PauliOperator[]
    sizehint!(ops, 4^k)
    
    # Enumerate all 4^k logical Pauli operators
    # Each index idx encodes a k-digit base-4 number
    for idx in 0:(4^k - 1)
        P = zero(PauliOperator, n)
        for i in 1:k
            digit = (idx >> (2*(i-1))) & 3
            if digit == 1      # X
                P = P * LX[i]
            elseif digit == 2  # Y = iXZ
                P = P * LX[i] * LZ[i]
                # Y = iXZ, so multiply by i (add 1 to phase)
                P.phase[] = (P.phase[] + 0x1) & 0x3
            elseif digit == 3  # Z
                P = P * LZ[i]
            end
            # digit == 0 is identity, do nothing
        end
        push!(ops, P)
    end
    return ops
end

"""
    find_minimal_support_representative(L::PauliOperator, stab::MixedDestabilizer, A::Vector{Int}) -> Union{PauliOperator, Nothing}

Find a representative of the logical operator L (modulo stabilizers) that is 
supported only on subsystem A, if one exists.

This uses Gaussian elimination on the stabilizer group restricted to the complement 
of A to "clean" the logical operator.

# Algorithm
1. Let Ā = complement of A (qubits outside A)
2. For each stabilizer generator S_i, check if multiplying L by S_i reduces support on Ā
3. Use Gaussian elimination to systematically remove support on Ā

# Returns
- `PauliOperator`: A representative supported on A, or
- `nothing`: If no such representative exists

"""
function find_minimal_support_representative(L::PauliOperator, stab::MixedDestabilizer, A::Vector{Int})
    n = nqubits(L)
    Abar = setdiff(1:n, A)  # Complement of A
    
    if isempty(Abar)
        # A is the full system, L is trivially supported on A
        return copy(L)
    end
    
    S = stabilizerview(stab)
    r = length(S)
    
    if r == 0
        # No stabilizers to multiply
        return is_supported_on(L, A) ? copy(L) : nothing
    end
    
    # We want to solve: find subset T ⊆ {1,...,r} such that 
    # L * ∏_{i∈T} S_i is supported on A
    # 
    # This is a system of linear equations over GF(2):
    # For each qubit j ∈ Ā, we need the X and Z bits to be 0
    # 
    # Let's build the matrix: columns are stabilizers, rows are (X_j, Z_j) for j ∈ Ā
    # Right-hand side is the corresponding bits of L
    
    num_constraints = 2 * length(Abar)
    
    # Build augmented matrix [M | b] over GF(2)
    M = zeros(Bool, num_constraints, r + 1)  # Last column is RHS
    
    for (row_idx, j) in enumerate(Abar)
        # X bit constraint for qubit j
        x_row = 2 * row_idx - 1
        # Z bit constraint for qubit j
        z_row = 2 * row_idx
        
        # Fill in stabilizer contributions
        for i in 1:r
            x_stab, z_stab = S[i][j]
            M[x_row, i] = x_stab
            M[z_row, i] = z_stab
        end
        
        # RHS from L
        x_L, z_L = L[j]
        M[x_row, r + 1] = x_L
        M[z_row, r + 1] = z_L
    end
    
    # Gaussian elimination over GF(2)
    pivot_row = 1
    pivot_cols = Int[]
    
    for col in 1:r
        # Find pivot
        found_pivot = false
        for row in pivot_row:num_constraints
            if M[row, col]
                # Swap rows
                if row != pivot_row
                    M[pivot_row, :], M[row, :] = M[row, :], M[pivot_row, :]
                end
                found_pivot = true
                break
            end
        end
        
        if !found_pivot
            continue
        end
        
        push!(pivot_cols, col)
        
        # Eliminate
        for row in 1:num_constraints
            if row != pivot_row && M[row, col]
                M[row, :] .⊻= M[pivot_row, :]
            end
        end
        
        pivot_row += 1
        if pivot_row > num_constraints
            break
        end
    end
    
    # Check for inconsistency: any row with all zeros in stabilizer columns but 1 in RHS
    for row in 1:num_constraints
        if !any(M[row, 1:r]) && M[row, r + 1]
            # No solution exists
            return nothing
        end
    end
    
    # Back-substitute to find which stabilizers to multiply
    solution = zeros(Bool, r)
    for (idx, col) in enumerate(pivot_cols)
        solution[col] = M[idx, r + 1]
    end
    
    # Construct the representative
    result = copy(L)
    for i in 1:r
        if solution[i]
            result = result * S[i]
        end
    end
    
    return result
end

"""
    subregion_logical_paulis(stab::MixedDestabilizer, A::Vector{Int}) -> Vector{PauliOperator}

Get all logical Pauli operators that have a representative supported only on subsystem A.

This function finds, for each of the 4^k logical operator equivalence classes, 
whether there exists a representative (logical operator times stabilizers) that 
is fully supported on the subsystem A. These operators form the subalgebra M_A ⊂ M.

# Algorithm
For each bare logical operator L:
1. Find if there exists stabilizers S₁, S₂, ... such that L * S₁ * S₂ * ... is supported on A
2. This is done via Gaussian elimination over GF(2)

# Arguments
- `stab::MixedDestabilizer`: The stabilizer code
- `A::Vector{Int}`: Qubit indices defining subsystem A

# Returns
- `Vector{PauliOperator}`: Representatives of logical operator classes supported on A

# Example
```julia
julia> s = S"XXXX
             ZIZI
             IZIZ"  # [[4,1,2]] code
julia> stab = MixedDestabilizer(s)
julia> sub_ops = subregion_logical_paulis(stab, [1, 2])
julia> P"ZZ__" in sub_ops  # Should find ZZ on qubits 1,2
true
```
"""
function subregion_logical_paulis(stab::MixedDestabilizer, A::Vector{Int})
    bare_logicals = all_logical_paulis(stab)
    
    result = PauliOperator[]
    for L in bare_logicals
        rep = find_minimal_support_representative(L, stab, A)
        if rep !== nothing
            push!(result, rep)
        end
    end
    
    return result
end

#=============================================================================
    Pauli Expectation Value Computation
=============================================================================#

"""
    pauli_to_matrix(P::PauliOperator) -> Matrix{ComplexF64}

Convert a Pauli operator to its 2^n × 2^n matrix representation.

# Arguments
- `P::PauliOperator`: n-qubit Pauli operator

# Returns  
- `Matrix{ComplexF64}`: The 2^n × 2^n matrix representation
"""
function pauli_to_matrix(P::PauliOperator)
    n = nqubits(P)
    
    # Single qubit Pauli matrices
    I2 = ComplexF64[1 0; 0 1]
    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    Z = ComplexF64[1 0; 0 -1]
    
    # Build tensor product
    mat = ComplexF64[1.0+0im;;]  # Start with 1×1 identity
    for i in 1:n
        x, z = P[i]
        if !x && !z
            local_mat = I2
        elseif x && !z
            local_mat = X
        elseif x && z
            local_mat = Y
        else  # !x && z
            local_mat = Z
        end
        mat = kron(mat, local_mat)
    end
    
    # Apply phase: 0x0=+1, 0x1=+i, 0x2=-1, 0x3=-i
    phase = P.phase[]
    phase_factor = im^phase
    return phase_factor * mat
end

"""
    stabilizer_state_vector(S::Stabilizer) -> Vector{ComplexF64}

Compute the state vector representation of a stabilizer state.
The stabilizer state |ψ⟩ is the unique state (up to phase) satisfying g|ψ⟩ = |ψ⟩ for all g ∈ S.

# Warning
This function has exponential complexity in the number of qubits and should only be used
for small systems (n ≤ 15).
"""
function stabilizer_state_vector(S::Stabilizer)
    n = nqubits(S)
    dim = 2^n
    
    # Project onto +1 eigenspace of all stabilizers
    projector = Matrix{ComplexF64}(I, dim, dim)
    
    for i in 1:length(S)
        P = S[i]
        P_mat = pauli_to_matrix(P)
        # Projector onto +1 eigenspace: (I + P)/2
        projector = projector * (Matrix{ComplexF64}(I, dim, dim) + P_mat) / 2
    end
    
    # The state is in the image of the projector
    # Find a non-zero column
    for j in 1:dim
        col = projector[:, j]
        norm_col = norm(col)
        if norm_col > 1e-10
            return col / norm_col
        end
    end
    
    error("Failed to find stabilizer state vector")
end

"""
    is_in_stabilizer_group(P::PauliOperator, stab::MixedDestabilizer) -> Tuple{Bool, ComplexF64}

Check if Pauli operator P is in the stabilizer group, and if so, return its phase.

Uses the rowdecompose-like approach: if P can be written as a product of stabilizers,
it's in the group.

# Returns
- `(true, phase)` if P is in the stabilizer group with the given phase factor
- `(false, 0)` if P is not in the stabilizer group
"""
function is_in_stabilizer_group(P::PauliOperator, stab::MixedDestabilizer)
    n = nqubits(P)
    S = stabilizerview(stab)
    r = length(S)
    
    # Try to express P as a product of stabilizer generators
    # This is equivalent to solving a linear system over GF(2)
    
    # Build the matrix of stabilizer generators (in GF(2))
    # Each stabilizer contributes a column, each qubit contributes 2 rows (X and Z bits)
    M = zeros(Bool, 2n, r + 1)  # Last column is for P
    
    for j in 1:r
        for i in 1:n
            x, z = S[j][i]
            M[i, j] = x
            M[n + i, j] = z
        end
    end
    
    # RHS: P
    for i in 1:n
        x, z = P[i]
        M[i, r + 1] = x
        M[n + i, r + 1] = z
    end
    
    # Gaussian elimination
    pivot_row = 1
    pivot_cols = Int[]
    
    for col in 1:r
        found_pivot = false
        for row in pivot_row:2n
            if M[row, col]
                if row != pivot_row
                    M[pivot_row, :], M[row, :] = M[row, :], M[pivot_row, :]
                end
                found_pivot = true
                break
            end
        end
        
        if !found_pivot
            continue
        end
        
        push!(pivot_cols, col)
        
        for row in 1:2n
            if row != pivot_row && M[row, col]
                M[row, :] .⊻= M[pivot_row, :]
            end
        end
        
        pivot_row += 1
        if pivot_row > 2n
            break
        end
    end
    
    # Check for inconsistency
    for row in 1:2n
        if !any(M[row, 1:r]) && M[row, r + 1]
            return (false, zero(ComplexF64))
        end
    end
    
    # P is in the group. Compute the phase.
    solution = zeros(Bool, r)
    for (idx, col) in enumerate(pivot_cols)
        if idx <= length(pivot_cols)
            solution[col] = M[idx, r + 1]
        end
    end
    
    # Reconstruct the product and compute phase
    accumulated_phase = 0x0
    result = zero(PauliOperator, n)
    for i in 1:r
        if solution[i]
            # Multiply result by S[i], tracking phase
            result = result * S[i]
        end
    end
    
    # The phase of the product should match P's phase for P to be in the group
    # Actually, we need to check if result equals P (including phase)
    # result and P should have the same X,Z bits (guaranteed by the linear algebra)
    # The phase of result is accumulated_phase, phase of P is P.phase[]
    
    # The eigenvalue is i^(P.phase[] - result.phase[]) = i^diff
    diff = (P.phase[] - result.phase[]) & 0x3
    phase_factor = im^diff
    
    return (true, phase_factor)
end

"""
    expect_pauli(P::PauliOperator, S::Stabilizer) -> ComplexF64

Compute the expectation value ⟨ψ|P|ψ⟩ for a Pauli operator P on stabilizer state |ψ⟩.

**Efficient Implementation using Stabilizer Formalism:**

For a Pauli operator P on a stabilizer state |ψ⟩ defined by stabilizer group S:

1. If P anticommutes with any stabilizer generator: ⟨ψ|P|ψ⟩ = 0
2. If P is in the stabilizer group S: ⟨ψ|P|ψ⟩ = ±1 or ±i (determined by the phase)
3. If P commutes with S but is not in S: ⟨ψ|P|ψ⟩ = 0 (P is a non-trivial logical operator)

This implementation has complexity O(n³) where n is the number of qubits,
avoiding the exponential cost of state vector computation.

# Arguments
- `P::PauliOperator`: The Pauli operator
- `S::Stabilizer`: The stabilizer state (must be full rank for pure state)

# Returns
- `ComplexF64`: The expectation value
"""
function expect_pauli(P::PauliOperator, S::Stabilizer)
    n = nqubits(S)
    r = length(S)
    
    # Step 1: Check if P anticommutes with any stabilizer
    for i in 1:r
        if comm(P, S[i]) != 0
            return zero(ComplexF64)
        end
    end
    
    # Step 2: P commutes with all stabilizers.
    # Check if P is in the stabilizer group.
    
    # For a full-rank stabilizer (pure state), if P commutes with all generators
    # and is not the identity, we need to check if it's in the group.
    
    # Use MixedDestabilizer for the decomposition
    stab = MixedDestabilizer(S)
    
    in_group, phase = is_in_stabilizer_group(P, stab)
    
    if in_group
        return phase
    else
        # P commutes with S but is not in S
        # This means P is a logical operator (for codes) or something else
        # For a pure stabilizer state, expectation is 0
        return zero(ComplexF64)
    end
end

"""
    expect_pauli(P::PauliOperator, stab::MixedDestabilizer) -> ComplexF64

Compute ⟨ψ|P|ψ⟩ for a Pauli operator on a state represented by MixedDestabilizer.
"""
function expect_pauli(P::PauliOperator, stab::MixedDestabilizer)
    S = stabilizerview(stab)
    r = length(S)
    
    # Check anticommutation
    for i in 1:r
        if comm(P, S[i]) != 0
            return zero(ComplexF64)
        end
    end
    
    # Check if in stabilizer group
    in_group, phase = is_in_stabilizer_group(P, stab)
    
    if in_group
        return phase
    else
        return zero(ComplexF64)
    end
end

#=============================================================================
    Logical Density Matrix and Entropy Computation
=============================================================================#

"""
    stabilizer_inner_product_with_pauli(ψ1::Stabilizer, P::PauliOperator, ψ2::Stabilizer) -> ComplexF64

Compute ⟨ψ1|P|ψ2⟩ for two stabilizer states and a Pauli operator.

Uses the fact that P|ψ2⟩ is also a stabilizer state (up to phase), 
and the inner product of two stabilizer states is efficiently computable.
"""
function stabilizer_inner_product_with_pauli(ψ1::Stabilizer, P::PauliOperator, ψ2::Stabilizer)
    n = nqubits(P)
    @assert nqubits(ψ1) == n && nqubits(ψ2) == n
    
    # For small systems, use explicit state vectors
    if n <= 12
        v1 = stabilizer_state_vector(ψ1)
        v2 = stabilizer_state_vector(ψ2)
        P_mat = pauli_to_matrix(P)
        return dot(v1, P_mat * v2)
    else
        error("Large system inner product not yet implemented")
    end
end

"""
    paulis_equivalent(P1::PauliOperator, P2::PauliOperator) -> Bool

Check if two Pauli operators have the same X and Z bits (ignoring phase).
"""
function paulis_equivalent(P1::PauliOperator, P2::PauliOperator)
    n = nqubits(P1)
    nqubits(P2) == n || return false
    for i in 1:n
        P1[i] == P2[i] || return false
    end
    return true
end

"""
    logical_pauli_to_matrix(P::PauliOperator, basis::Vector{PauliOperator}, dim::Int) -> Matrix{ComplexF64}

Convert a physical Pauli operator (representing a logical operator) to its 
logical qubit matrix representation.
"""
function logical_pauli_to_matrix(P::PauliOperator, basis::Vector{PauliOperator}, dim::Int)
    n_logical = round(Int, log2(dim))
    
    # Single qubit Pauli matrices
    I2 = ComplexF64[1 0; 0 1]
    X = ComplexF64[0 1; 1 0]  
    Y = ComplexF64[0 -im; im 0]
    Z = ComplexF64[1 0; 0 -1]
    
    # Find which basis element P corresponds to
    idx = findfirst(B -> paulis_equivalent(P, B), basis)
    if isnothing(idx)
        return zeros(ComplexF64, dim, dim)
    end
    
    idx -= 1  # Convert to 0-based
    
    # Build tensor product based on idx encoding
    mat = ComplexF64[1.0+0im;;]
    for i in 1:n_logical
        digit = (idx >> (2*(i-1))) & 3
        local_mat = if digit == 0
            I2
        elseif digit == 1
            X
        elseif digit == 2
            Y
        else
            Z
        end
        mat = kron(mat, local_mat)
    end
    
    # Apply phase from P
    phase_factor = im^(P.phase[])
    return phase_factor * mat
end

"""
    compute_logical_density_matrix(S::Stabilizer, A::Vector{Int}, 
                                   states::Vector{<:Stabilizer}, 
                                   coeffs::Vector{<:Number}) -> Matrix{ComplexF64}

Compute the logical density matrix ρ_A for subsystem A.

Given a state |Ψ⟩ = Σ_j c_j |ψ_j⟩ where |ψ_j⟩ are stabilizer states in the code space
defined by S, compute the reduced density matrix on the logical subspace accessible
from region A.

The density matrix is computed via Pauli tomography:
    ρ_A = (1/d) Σ_P ⟨Ψ|P|Ψ⟩ P

where P ranges over all logical Paulis supported on A, and d = dim(M_A).

# Arguments
- `S::Stabilizer`: The common stabilizer group
- `A::Vector{Int}`: Subsystem qubit indices
- `states::Vector{Stabilizer}`: The stabilizer states |ψ_j⟩
- `coeffs::Vector{Number}`: The coefficients c_j

# Returns
- `Matrix{ComplexF64}`: The logical density matrix ρ_A
"""
function compute_logical_density_matrix(S::Stabilizer, A::Vector{Int}, 
                                        states::Vector{<:Stabilizer}, 
                                        coeffs::Vector{<:Number})
    n = nqubits(S)
    K = length(states)
    @assert K == length(coeffs) "Number of states must match number of coefficients"
    @assert all(1 .≤ A .≤ n) "Subsystem A indices out of bounds"
    
    # Get logical operators supported on A
    stab_md = MixedDestabilizer(S)
    sub_logical = subregion_logical_paulis(stab_md, A)
    d = length(sub_logical)
    
    if d == 0
        error("No logical operators supported on subsystem A")
    end
    
    # Dimension: d = 4^{k_A}, so dim = 2^{k_A}
    k_a = round(Int, log2(d) / 2)
    dim = 2^k_a
    
    # Compute expectation values ⟨Ψ|P|Ψ⟩ for each logical Pauli P
    expvals = ComplexF64[]
    for P in sub_logical
        ev = zero(ComplexF64)
        for α in 1:K, β in 1:K
            ov = stabilizer_inner_product_with_pauli(states[α], P, states[β])
            ev += conj(coeffs[α]) * coeffs[β] * ov
        end
        push!(expvals, ev)
    end
    
    # Reconstruct density matrix
    ρ_a = zeros(ComplexF64, dim, dim)
    for (i, P) in enumerate(sub_logical)
        P_logical_mat = logical_pauli_to_matrix(P, sub_logical, dim)
        ρ_a .+= (expvals[i] / d) .* P_logical_mat
    end
    
    # Ensure Hermitian and normalized
    ρ_a = (ρ_a + ρ_a') / 2
    if abs(tr(ρ_a)) > 1e-10
        ρ_a ./= tr(ρ_a)
    end
    
    return ρ_a
end

"""
    compute_logical_entropy(S::Stabilizer, A::Vector{Int}, 
                           states::Vector{<:Stabilizer}, 
                           coeffs::Vector{<:Number}) -> Float64

Compute the von Neumann entropy S(ρ_A) of the logical density matrix.

This implements Algorithm 1 from the paper "An efficient algorithm to compute 
entanglement in states with low magic" (arXiv:2510.06318).

# Algorithm Overview
1. Generate all 4^k logical Pauli operators from the stabilizer code
2. Filter to keep only those supported on subsystem A (forming M_A)  
3. Compute expectation values ⟨Ψ|P|Ψ⟩ for each P ∈ M_A
4. Reconstruct ρ_A via Pauli tomography
5. Diagonalize and compute entropy

# Arguments
- `S::Stabilizer`: The common stabilizer group defining the code
- `A::Vector{Int}`: Qubit indices defining subsystem A
- `states::Vector{Stabilizer}`: The stabilizer states |ψ_j⟩ in superposition
- `coeffs::Vector{Number}`: Coefficients c_j where |Ψ⟩ = Σ c_j |ψ_j⟩

# Returns
- `Float64`: The von Neumann entropy S(ρ_A) = -Tr(ρ_A log₂ ρ_A)
"""
function compute_logical_entropy(S::Stabilizer, A::Vector{Int}, 
                                 states::Vector{<:Stabilizer}, 
                                 coeffs::Vector{<:Number})
    ρ_A = compute_logical_density_matrix(S, A, states, coeffs)
    
    # Diagonalize
    λ = eigvals(Hermitian(ρ_A))
    
    # Compute von Neumann entropy
    entropy = 0.0
    for v in λ
        if v > 1e-15
            entropy -= v * log2(v)
        end
    end
    
    return entropy
end

# Convenience method for single stabilizer state
function compute_logical_entropy(S::Stabilizer, A::Vector{Int}, state::Stabilizer)
    return compute_logical_entropy(S, A, [state], [1.0])
end
