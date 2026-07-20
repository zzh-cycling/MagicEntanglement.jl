using QuantumClifford
using QuantumClifford: MixedDestabilizer, Stabilizer, PauliOperator, nqubits,
                       stab_to_gf2, stabilizerview, logicalzview
using LinearAlgebra

#=============================================================================
    GF(2) Linear Algebra Helpers
=============================================================================#

"""
    gf2_rank(M::AbstractMatrix{Bool}) -> Int

Compute the rank of a binary matrix over GF(2) via Gaussian elimination.
"""
function gf2_rank(M::AbstractMatrix{Bool})
    A = copy(M)
    nrows, ncols = size(A)
    rank = 0
    pivot_row = 1
    for col in 1:ncols
        pivot = findfirst(i -> A[i, col], pivot_row:nrows)
        isnothing(pivot) && continue
        pivot += pivot_row - 1
        if pivot != pivot_row
            A[pivot_row, :], A[pivot, :] = A[pivot, :], A[pivot_row, :]
        end
        for row in 1:nrows
            if row != pivot_row && A[row, col]
                A[row, :] .⊻= A[pivot_row, :]
            end
        end
        rank += 1
        pivot_row += 1
        pivot_row > nrows && break
    end
    return rank
end

#=============================================================================
    Stabilizer State Entanglement (Fattal et al., quant-ph/0406168)
=============================================================================#

"""
    stabilizer_entropy(S::Stabilizer, A::Vector{Int}) -> Float64

Compute the bipartite von Neumann entanglement entropy S(ρ_A) (in bits) of a
pure stabilizer state across the bipartition A | Ā.

Uses the formula S(ρ_A) = |A| - log₂|S_A|, where S_A is the subgroup of the
stabilizer group whose elements are supported entirely on A. The size of S_A
is obtained from a GF(2) rank computation, giving O(n³) complexity.

# Example
```julia
julia> bell = S"XX
                ZZ"
julia> stabilizer_entropy(bell, [1])
1.0
```
"""
function stabilizer_entropy(S::Stabilizer, A::Vector{Int})
    n = nqubits(S)
    r = length(S)
    Abar = setdiff(1:n, A)

    # Binary symplectic form: rows are generators, columns are [x_1..x_n | z_1..z_n]
    gf2 = stab_to_gf2(S)

    # Restrict generators to Ā; the stabilizers supported on A form the kernel
    # of this restriction map, so log₂|S_A| = r - rank(M_Ā)
    M = zeros(Bool, 2 * length(Abar), r)
    for (row_idx, j) in enumerate(Abar)
        M[2 * row_idx - 1, :] .= gf2[:, j]          # x bit of qubit j
        M[2 * row_idx, :]     .= gf2[:, n + j]      # z bit of qubit j
    end

    log2_SA = r - gf2_rank(M)
    return Float64(length(A) - log2_SA)
end

#=============================================================================
    Full Algorithm 1 of arXiv:2510.06318
=============================================================================#

"""
    reference_stabilizer_state(S::Stabilizer) -> Stabilizer

Construct a reference stabilizer state |ϕ⟩ in the code subspace of S by
extending the stabilizer generators with the logical Z operators:
|ϕ⟩ is stabilized by S ∪ {Z̄_i}. Used to compute the state-independent
area term of the RT decomposition (Theorem 1 of arXiv:2510.06318).
"""
function reference_stabilizer_state(S::Stabilizer)
    md = MixedDestabilizer(S)
    return vcat(stabilizerview(md), Stabilizer(logicalzview(md)))
end

"""
    compute_entanglement(S::Stabilizer, A::Vector{Int},
                         states::Vector{<:Stabilizer},
                         coeffs::Vector{<:Number}) -> Float64

Compute the von Neumann entanglement entropy S(ρ_A) (in bits) of the state
|Ψ⟩ = Σ_j c_j |ψ_j⟩ across the bipartition A | Ā, where the |ψ_j⟩ share the
common stabilizer group S of rank n - ν.

This implements the full Algorithm 1 of "An efficient algorithm to compute
entanglement in states with low magic" (arXiv:2510.06318):

  S(ρ_A) = S(ρ_a) + 𝒜

where S(ρ_a) is the logical (bulk) entropy from `compute_logical_entropy`,
and the area term 𝒜 = S(ϕ_A) - S(ϕ_a) is state-independent (Theorem 1), so it
is evaluated with a reference stabilizer state |ϕ⟩ stabilized by S ∪ {Z̄_i}.

# Returns
- `Float64`: S(ρ_A) in bits.
"""
function compute_entanglement(S::Stabilizer, A::Vector{Int},
                              states::Vector{<:Stabilizer},
                              coeffs::Vector{<:Number})
    s_bulk = compute_logical_entropy(S, A, states, coeffs)

    ϕ = reference_stabilizer_state(S)
    s_ref_total = stabilizer_entropy(ϕ, A)
    s_ref_bulk = compute_logical_entropy(S, A, [ϕ], [1.0])
    area = s_ref_total - s_ref_bulk

    return s_bulk + area
end

# Convenience method for a single stabilizer state
function compute_entanglement(S::Stabilizer, A::Vector{Int}, state::Stabilizer)
    return compute_entanglement(S, A, [state], [1.0])
end

"""
    compute_renyi_entropy(n_renyi::Integer, S::Stabilizer, A::Vector{Int},
                          states::Vector{<:Stabilizer},
                          coeffs::Vector{<:Number}) -> Float64

Compute the Rényi-n entanglement entropy S_n(ρ_A) (in bits) of
|Ψ⟩ = Σ_j c_j |ψ_j⟩ across A | Ā.

Uses the Rényi RT formula (Appendix B of arXiv:2510.06318):

  S_n(ρ_A) = S_n(ρ_a) + 𝒜

with the same state-independent area term 𝒜 as in the von Neumann case
(the stabilizer contribution has a flat spectrum, hence n-independent).
"""
function compute_renyi_entropy(n_renyi::Integer, S::Stabilizer, A::Vector{Int},
                               states::Vector{<:Stabilizer},
                               coeffs::Vector{<:Number})
    n_renyi >= 2 || throw(ArgumentError("use compute_entanglement for n = 1"))

    ρ_a = compute_logical_density_matrix(S, A, states, coeffs)
    λ = eigvals(Hermitian(ρ_a))
    s_bulk = log2(sum(v -> v^n_renyi, filter(v -> v > 1e-15, λ))) / (1 - n_renyi)

    ϕ = reference_stabilizer_state(S)
    area = stabilizer_entropy(ϕ, A) - compute_logical_entropy(S, A, [ϕ], [1.0])

    return s_bulk + area
end

"""
    entanglement_spectrum(S::Stabilizer, A::Vector{Int},
                          states::Vector{<:Stabilizer},
                          coeffs::Vector{<:Number}) -> Vector{Float64}

Compute the entanglement spectrum (eigenvalues of ρ_A) of |Ψ⟩ = Σ_j c_j |ψ_j⟩
across A | Ā.

Per Eq. (B7) of arXiv:2510.06318, the spectrum consists of the eigenvalues λ
of the logical density matrix ρ_a, each scaled and degenerate:
every λ appears with multiplicity d_χ = 2^𝒜 as λ/d_χ, where 𝒜 is the area term.
"""
function entanglement_spectrum(S::Stabilizer, A::Vector{Int},
                               states::Vector{<:Stabilizer},
                               coeffs::Vector{<:Number})
    ρ_a = compute_logical_density_matrix(S, A, states, coeffs)
    λ = eigvals(Hermitian(ρ_a))

    ϕ = reference_stabilizer_state(S)
    area = stabilizer_entropy(ϕ, A) - compute_logical_entropy(S, A, [ϕ], [1.0])
    dχ = round(Int, 2^area)

    spectrum = Float64[]
    sizehint!(spectrum, length(λ) * dχ)
    for v in λ
        v > 1e-15 || continue
        append!(spectrum, fill(v / dχ, dχ))
    end
    return sort!(spectrum, rev=true)
end
