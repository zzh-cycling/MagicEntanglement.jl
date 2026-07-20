using MagicEntanglement
using QuantumClifford
using LinearAlgebra
using Test

# Brute-force reference: entanglement entropy of a state vector across A | Ā
function brute_force_entropy(ψ::Vector{<:Number}, A::Vector{Int}, n::Int)
    Abar = setdiff(1:n, A)
    perm = vcat(A, Abar)
    dA, dAbar = 2^length(A), 2^length(Abar)
    M = reshape(permutedims(reshape(ψ, fill(2, n)...), perm), dA, dAbar)
    ρ_A = M * M'
    λ = eigvals(Hermitian(ρ_A))
    return -sum(v -> v > 1e-12 ? v * log2(v) : 0.0, λ)
end

@testset "gf2_rank" begin
    @test gf2_rank(Bool[1 0; 0 1]) == 2
    @test gf2_rank(Bool[1 1; 1 1]) == 1
    @test gf2_rank(Bool[0 0; 0 0]) == 0
    @test gf2_rank(Bool[1 0 1; 0 1 1; 1 1 0]) == 2
end

@testset "stabilizer_entropy" begin
    # Bell state: 1 ebit across any nontrivial bipartition
    bell = S"XX
             ZZ"
    @test stabilizer_entropy(bell, [1]) ≈ 1.0
    @test stabilizer_entropy(bell, [2]) ≈ 1.0
    @test stabilizer_entropy(bell, [1, 2]) ≈ 0.0

    # Product state |00⟩
    prod2 = S"ZI
              IZ"
    @test stabilizer_entropy(prod2, [1]) ≈ 0.0

    # GHZ state: 1 bit for any single qubit
    ghz = S"XXX
            ZZI
            ZIZ"
    @test stabilizer_entropy(ghz, [1]) ≈ 1.0
    @test stabilizer_entropy(ghz, [1, 2]) ≈ 1.0

    # Cluster-like entangled state on 4 qubits
    s4 = S"XXXX
           ZZII
           IZZI
           IIZZ"
    @test stabilizer_entropy(s4, [1, 2]) ≈ brute_force_entropy(stabilizer_state_vector(s4), [1, 2], 4)
end

@testset "reference_stabilizer_state" begin
    # [[4,1,2]] code: reference state must be a full-rank stabilizer state in the code
    s = S"XXXX
          ZIZI
          IZIZ"
    ϕ = reference_stabilizer_state(s)
    @test length(ϕ) == 4
    # Stabilized by the original code stabilizers
    for i in 1:3
        @test real(expect_pauli(s[i], ϕ)) ≈ 1.0 atol=1e-10
    end
end

@testset "compute_entanglement - paper example Eq.(6)-(12)" begin
    # [[4,1,2]] code, A = {1,2}; |ψ⟩ = c1|s1⟩ + c2|s2⟩ with
    # |s1⟩ = (|0101⟩+|1010⟩)/√2, |s2⟩ = (|0000⟩+|1111⟩)/√2
    S_code = S"XXXX
               ZIZI
               IZIZ"
    s1 = S"XXXX
           ZIZI
           IZIZ
           -ZZII"
    s2 = S"XXXX
           ZIZI
           IZIZ
           ZZII"
    A = [1, 2]

    # Equal superposition: ρ_A = I/4, S = 2 bits
    c = 1 / sqrt(2)
    @test compute_entanglement(S_code, A, [s1, s2], [c, c]) ≈ 2.0 atol=1e-8

    # Pure stabilizer limits: 1 ebit
    @test compute_entanglement(S_code, A, [s1, s2], [1.0, 0.0]) ≈ 1.0 atol=1e-8
    @test compute_entanglement(S_code, A, [s1, s2], [0.0, 1.0]) ≈ 1.0 atol=1e-8

    # General coefficients vs brute force statevector calculation
    for (c1, c2) in [(sqrt(0.3), sqrt(0.7)), (sqrt(0.5), im*sqrt(0.5)), (sqrt(0.9), sqrt(0.1))]
        ψ = c1 * stabilizer_state_vector(s1) + c2 * stabilizer_state_vector(s2)
        ψ ./= norm(ψ)
        expected = brute_force_entropy(ψ, A, 4)
        @test compute_entanglement(S_code, A, [s1, s2], [c1, c2]) ≈ expected atol=1e-8
    end
end

@testset "compute_entanglement - 5-qubit [[5,2]] code (Appendix C)" begin
    # Common stabilizer group S = ⟨XZZXI, IXZZX, XIXZZ⟩, A = {1,2,3}
    S_code = S"XZZXI
               IXZZX
               XIXZZ"
    # Codewords: extend with two independent logical Z operators.
    # From Appendix C, Z̄1 = ZXZII-like representatives; here we use the
    # canonical logical Zs from the package to build codewords.
    md = MixedDestabilizer(S_code)
    LZ = QuantumClifford.logicalzview(md)
    S_ext = vcat(QuantumClifford.stabilizerview(md), Stabilizer(LZ))
    A = [1, 2, 3]

    # Pure codeword: entropy must equal the area term and match brute force
    ϕ = Stabilizer(S_ext)
    s_ref = compute_entanglement(S_code, A, [ϕ], [1.0])
    expected = brute_force_entropy(stabilizer_state_vector(ϕ), A, 5)
    @test s_ref ≈ expected atol=1e-8
    @test stabilizer_entropy(ϕ, A) ≈ expected atol=1e-8
end

@testset "compute_renyi_entropy" begin
    S_code = S"XXXX
               ZIZI
               IZIZ"
    s1 = S"XXXX
           ZIZI
           IZIZ
           -ZZII"
    s2 = S"XXXX
           ZIZI
           IZIZ
           ZZII"
    A = [1, 2]

    # Maximally entangled case: flat spectrum, all Rényi indices give 2 bits
    c = 1 / sqrt(2)
    for n in 2:5
        @test compute_renyi_entropy(n, S_code, A, [s1, s2], [c, c]) ≈ 2.0 atol=1e-8
    end

    # Rényi-2 for a non-flat case: ρ_A eigenvalues {0.7/2, 0.7/2, 0.3/2, 0.3/2}
    c1, c2 = sqrt(0.7), sqrt(0.3)
    λA = [0.35, 0.35, 0.15, 0.15]
    expected_S2 = -log2(sum(λA .^ 2))
    @test compute_renyi_entropy(2, S_code, A, [s1, s2], [c1, c2]) ≈ expected_S2 atol=1e-8
end

@testset "entanglement_spectrum" begin
    S_code = S"XXXX
               ZIZI
               IZIZ"
    s1 = S"XXXX
           ZIZI
           IZIZ
           -ZZII"
    s2 = S"XXXX
           ZIZI
           IZIZ
           ZZII"
    A = [1, 2]

    # Maximally entangled: spectrum {1/4, 1/4, 1/4, 1/4}
    c = 1 / sqrt(2)
    spec = entanglement_spectrum(S_code, A, [s1, s2], [c, c])
    @test length(spec) == 4
    @test all(v -> v ≈ 0.25, spec)
    @test sum(spec) ≈ 1.0 atol=1e-10

    # Non-flat case: degeneracy d_χ = 2 from the area term
    c1, c2 = sqrt(0.7), sqrt(0.3)
    spec = entanglement_spectrum(S_code, A, [s1, s2], [c1, c2])
    @test spec ≈ [0.35, 0.35, 0.15, 0.15] atol=1e-8

    # Consistency: -Tr ρ log₂ ρ reproduces compute_entanglement
    s_from_spec = -sum(v -> v * log2(v), spec)
    @test s_from_spec ≈ compute_entanglement(S_code, A, [s1, s2], [c1, c2]) atol=1e-8
end
