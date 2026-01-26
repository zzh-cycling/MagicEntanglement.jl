function compute_logical_entropy(S::Stabilizer, subsystems::Vector{Int}, ψ::Vector{ET}) where ET
    tableaux = MixedDestabilizer(S)
    logical_ops_X = logicalxview(tableaux)
    logical_ops_Z = logicalzview(tableaux)
    normalizer_ops = normalizer(tab(S))
   
    sub_logical_ops = [op for op in logical_ops if all(q in subsystems for q in qubits(op))]
    
    # Calculate entropy
    entropy = 0.0
    for op in sub_logical_ops
        p = abs2(dot(ψ, apply(op, ψ)))
        if p > 0
            entropy -= p * log2(p)
        end
    end
    
    return entropy
    
end

"""
    compute_logical_entropy(S::Stabilizer, A::Vector{Int}, Ψ::Vector{ET}) where ET<:Number

Full implementation of Algorithm 1 in the paper.
Input
    S   : Common stabilizer group (n physical qubits)
    A   : Subsystem qubit indices, e.g. [1,2,3]
    Ψ   : Coefficient vector of the low-magic state |Ψ⟩=Σ cⱼ|ψⱼ⟩, length must equal the number K of |ψⱼ⟩
Output
    S(ρ_a) : von Neumann entropy of logical subsystem a
"""
function compute_logical_entropy(S::Stabilizer, A::Vector{Int}, Ψ::Vector{ET}) where ET<:Number
    n = nqubits(S)               # Number of physical qubits
    @assert all(1 .≤ A .≤ n) "subsystem A out of bounds"
    K = length(Ψ)                # Number of superposition terms
    @assert K ≥ 1

    # ---------- 1. Obtain the complete logical Pauli basis ----------
    tab = MixedDestabilizer(S)   # Provides both stabilizer + logical
    k = rank(tab)                # Number of logical qubits k = ν
    # All 4^k logical Pauli operators (including I)
    logical_basis = all_logical_paulis(tab)   # Vector length 4^k

    # ---------- 2. Cleaning: keep only those fully supported on A ----------
    sub_logical = PauliOperator[]              # Will form the basis of M_A
    for P in logical_basis
        # Check if the support of P is ⊆ A
        support = [i for i in 1:n if !iszero(P,i)]
        if all(q -> q ∈ A, support)
            push!(sub_logical, P)
        end
    end
    d = length(sub_logical)        # Dimension of M_A (including I), must be 2^{k_a}
    k_a = round(Int, log2(d))      # Number of logical qubits recoverable from A
    @assert 2^k_a == d "internal dim error"

    # ---------- 3. Tomography on M_A to reconstruct ρ_a ----------
    ρ_a = zeros(ComplexF64, d, d)  # 2^{k_a} × 2^{k_a}
    # Map the basis of M_A to Pauli matrices in 2^{k_a}-dim Hilbert space
    # Standard encoding: the i-th Pauli corresponds to a 2^{k_a}×2^{k_a} matrix
    pauli_mat = Matrix{ComplexF64}[logical_matrix(p) for p in sub_logical]
    # Normalization: Tr(P_i P_j)=d δ_{ij}
    for (i, Pi) in enumerate(sub_logical)
        # Compute <Ψ|Pi|Ψ>
        expval = zero(ComplexF64)
        for α in 1:K, β in 1:K
            # |ψ_α⟩ is a stabilizer state, can use stabilizer inner product formula
            ov = dot(stab_state(S,α), apply(Pi, stab_state(S,β)))
            expval += conj(Ψ[α]) * Ψ[β] * ov
        end
        ρ_a .+= (expval / d) .* pauli_mat[i]
    end
    # Ensure Hermitian & trace=1
    ρ_a = (ρ_a + ρ_a') / 2
    ρ_a /= tr(ρ_a)

    # ---------- 4. Diagonalize to get entropy ----------
    λ = eigen!(Hermitian(ρ_a)).values
    S = 0.0
    for v in λ
        v > 0 && (S -= v * log2(v))
    end
    return S
end

# ---------- Generate all logical operators with type PauliOperator ----------
function all_logical_paulis(stab::MixedDestabilizer)
    r = rank(stab)
    n = nqubits(stab)
    k = n -r # number of logical qubits
    LX_tab = logicalxview(stab).tab
    LZ_tab = logicalzview(stab).tab
    stab_tab = tab(stab)

    ops = PauliOperator[LX_tab..., LZ_tab...]
    for i in 1:k
        for j in 1:r
            push!(ops, LX_tab[i]*stab_tab[j])
            push!(ops, LZ_tab[i]*stab_tab[j])
        end
    end
    for s in stab_tab
    end
    for idx in 0:(4^k-1)
        # Treat idx as k-digit base-4: 0=I,1=X,2=Y,3=Z
        P = one(PauliOperator, nqubits(stab))
        for i in 1:k
            digit = (idx >> (2(i-1))) & 3
            if digit == 1
                P *= LX_tab[i]
            elseif digit == 3
                P *= LZ_tab[i]
            elseif digit == 2
                P *= im * LX_tab[i] * LZ_tab[i]
            end
        end
        push!(ops, P)
    end
    ops
end

# ---------- Helper: fast inner product for stabilizer state ----------
function stab_state(S::Stabilizer, idx::Int)
    # Here we assume |ψ_idx⟩ is a certain logical computational basis state in the code subspace (e.g., |+⟩ or |0⟩).
    # For demonstration, we directly take the tableau representation of the idx-th stabilizer state.
    # In practice, |ψ_j⟩ can be pre-stored as Stabilizer objects.
    tab = copy(S)
    # Simple approach: set logical Z to +1 eigenstate
    logicalz = logicalzview(tab)
    for i in 1:length(logicalz)
        apply!(tab, logicalz[i])   # Measure and project
    end
    tab
end