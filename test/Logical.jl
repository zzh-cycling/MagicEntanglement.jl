using MagicEntanglement
using QuantumClifford   

@testset "all_logical_paulis" begin
    # Test with [[4, 1, 2]] code (4 physical qubits, 1 logical qubit)
    s = S"XXXX
          ZIZI
          IZIZ"
    stb = MixedDestabilizer(s)
    logicals = all_logical_paulis(stb)
    @test length(logicals) == 4  # 1 logical qubit -> 4^1 = 4 logical Paulis
    @test logicals == unique(logicals)  # no duplicates
end

@testset "pauli_support" begin
    @test pauli_support(P"_XY_Z") == [2, 3, 5]
    @test pauli_support(P"____") == Int[]
    @test pauli_support(P"XYZZ") == [1, 2, 3, 4]
end

@testset "is_supported_on" begin
    @test is_supported_on(P"_X__", [1, 2, 3])
    @test is_supported_on(P"_X__", [2])
    @test !is_supported_on(P"_X__", [1, 3])
    @test is_supported_on(P"____", [1])  # Identity supported everywhere
end

@testset "subregion_logical_paulis" begin
    s = S"XXXX
          ZIZI
          IZIZ"
    stb = MixedDestabilizer(s)
    
    # All logical operators should have support somewhere
    all_ops = all_logical_paulis(stb)
    @test length(all_ops) == 4
    
    # For full system, all should be included
    full_ops = subregion_logical_paulis(stb, [1, 2, 3, 4])
    @test length(full_ops) == 4
end

@testset "pauli_to_matrix" begin
    # Test single qubit Paulis
    X_mat = pauli_to_matrix(P"X")
    @test X_mat ≈ [0 1; 1 0]
    
    Z_mat = pauli_to_matrix(P"Z")
    @test Z_mat ≈ [1 0; 0 -1]
    
    Y_mat = pauli_to_matrix(P"Y")
    @test Y_mat ≈ [0 -im; im 0]
    
    I_mat = pauli_to_matrix(P"I")
    @test I_mat ≈ [1 0; 0 1]
    
    # Test phase
    mX_mat = pauli_to_matrix(P"-X")
    @test mX_mat ≈ -[0 1; 1 0]
end

@testset "stabilizer_state_vector" begin
    # |0⟩ state
    s0 = S"Z"
    v0 = stabilizer_state_vector(s0)
    @test abs(v0[1])^2 ≈ 1.0 || abs(v0[2])^2 ≈ 1.0
    
    # |+⟩ state
    sp = S"X"
    vp = stabilizer_state_vector(sp)
    @test abs(vp[1])^2 ≈ 0.5 atol=1e-10
    @test abs(vp[2])^2 ≈ 0.5 atol=1e-10
    
    # Bell state |00⟩ + |11⟩
    bell = S"XX
             ZZ"
    vbell = stabilizer_state_vector(bell)
    @test abs(vbell[1])^2 ≈ 0.5 atol=1e-10  # |00⟩
    @test abs(vbell[4])^2 ≈ 0.5 atol=1e-10  # |11⟩
end

@testset "expect_pauli" begin
    # |0⟩ state: ⟨Z⟩ = 1, ⟨X⟩ = 0
    s0 = S"Z"
    @test real(expect_pauli(P"Z", s0)) ≈ 1.0 atol=1e-10
    @test abs(expect_pauli(P"X", s0)) < 1e-10
    
    # |+⟩ state: ⟨X⟩ = 1, ⟨Z⟩ = 0
    sp = S"X"
    @test real(expect_pauli(P"X", sp)) ≈ 1.0 atol=1e-10
    @test abs(expect_pauli(P"Z", sp)) < 1e-10
end
