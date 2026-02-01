using QuantumClifford
using QuantumClifford: logicalxview, logicalzview, MixedDestabilizer, Stabilizer, 
                       PauliOperator, nqubits, rank, tab, stab_to_gf2, comm
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

The encoding is:
- 0 → I (identity)
- 1 → X (logical X)  
- 2 → Y (logical Y = iXZ)
- 3 → Z (logical Z)

# Arguments
- `stab::MixedDestabilizer`: A MixedDestabilizer representing the stabilizer code

# Returns
- `Vector{PauliOperator}`: All 4^k logical Pauli operators (including identity)

# Example
```julia
julia> s = S"XXXX+ZIZI+IZIZ"  # [[4,1,2]] code
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
    subregion_logical_paulis(stab::MixedDestabilizer, A::Vector{Int}) -> Vector{PauliOperator}

Get all logical Pauli operators that are fully supported on subsystem A.

This function filters the complete set of 4^k logical operators to return only those
whose support is contained within the specified subsystem A. These operators form
the subalgebra M_A ⊂ M of logical operators accessible from region A.

# Arguments
- `stab::MixedDestabilizer`: The stabilizer code
- `A::Vector{Int}`: Qubit indices defining subsystem A

# Returns
- `Vector{PauliOperator}`: Logical operators supported on A (forming M_A)

# Example
```julia
julia> s = S"XXXX+ZIZI+IZIZ"
julia> stab = MixedDestabilizer(s)
julia> sub_ops = subregion_logical_paulis(stab, [1, 2])
```
"""
function subregion_logical_paulis(stab::MixedDestabilizer, A::Vector{Int})
    all_ops = all_logical_paulis(stab)
    return filter(P -> is_supported_on(P, A), all_ops)
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
    expect_pauli(P::PauliOperator, S::Stabilizer) -> ComplexF64

Compute the expectation value ⟨ψ|P|ψ⟩ for a Pauli operator P on stabilizer state |ψ⟩.

For a Pauli operator P on a stabilizer state |ψ⟩:
- If P ∈ S (stabilizer group): ⟨ψ|P|ψ⟩ = ±1 (determined by phase)
- If P anticommutes with any stabilizer: ⟨ψ|P|ψ⟩ = 0
- Otherwise: more complex calculation needed

# Arguments
- `P::PauliOperator`: The Pauli operator
- `S::Stabilizer`: The stabilizer state

# Returns
- `ComplexF64`: The expectation value
"""
function expect_pauli(P::PauliOperator, S::Stabilizer)
    # Check if P commutes with all stabilizers
    for i in 1:length(S)
        Si = S[i]
        # comm(P, Si) returns 0 if they commute, 1 if they anticommute
        if comm(P, Si) != 0
            return zero(ComplexF64)  # Anticommutes => expectation is 0
        end
    end
    
    # P commutes with all stabilizers
    # For small systems, use matrix representation
    n = nqubits(S)
    if n <= 12
        ψ = stabilizer_state_vector(S)
        P_mat = pauli_to_matrix(P)
        return dot(ψ, P_mat * ψ)
    else
        error("Large system expectation not yet implemented efficiently")
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
