using MagicEntanglement
using QuantumClifford   
using Test

@testset "all_logical_paulis" begin
    # Test with [[4, 1, 2]] code (4 physical qubits, 1 logical qubit)
    s = S"XXXX
          ZIZI
          IZIZ"
    stb = MixedDestabilizer(s)
    logicals = all_logical_paulis(stb)
    @test length(logicals) == 4  # 1 logical qubit -> 4^1 = 4 logical Paulis
    @test logicals == unique(logicals)  # no duplicates
    
    # Note: all_logical_paulis returns bare representatives, not all equivalents
    # The logical Z may not be ZZ__ directly, but subregion_logical_paulis will find it
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

@testset "subregion_logical_paulis - Issue #3 test case" begin
    # This is the exact test case from Issue #3
    s = S"XXXX
          ZIZI
          IZIZ"
    stb = MixedDestabilizer(s)
    
    # Test that ZZ__ can be found as a logical operator supported on qubits [1,2]
    sub_ops = subregion_logical_paulis(stb, [1, 2])
    
    # Check that we can find an operator equivalent to ZZ__ (ignoring phase)
    function paulis_same_support(P1::PauliOperator, P2::PauliOperator)
        n = nqubits(P1)
        nqubits(P2) == n || return false
        for i in 1:n
            P1[i] == P2[i] || return false
        end
        return true
    end
    
    zz_found = any(paulis_same_support(op, P"ZZ__") for op in sub_ops)
    @test zz_found  # P"ZZ__" should be found as a logical op supported on [1,2]
    
    # The subregion should also contain identity
    @test length(sub_ops) >= 1
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

@testset "expect_pauli - efficient stabilizer formalism" begin
    # |0⟩ state: ⟨Z⟩ = 1, ⟨X⟩ = 0
    s0 = S"Z"
    @test real(expect_pauli(P"Z", s0)) ≈ 1.0 atol=1e-10
    @test abs(expect_pauli(P"X", s0)) < 1e-10
    
    # |+⟩ state: ⟨X⟩ = 1, ⟨Z⟩ = 0
    sp = S"X"
    @test real(expect_pauli(P"X", sp)) ≈ 1.0 atol=1e-10
    @test abs(expect_pauli(P"Z", sp)) < 1e-10
    
    # Bell state: ⟨XX⟩ = 1, ⟨ZZ⟩ = 1, ⟨XI⟩ = 0, ⟨ZI⟩ = 0
    bell = S"XX
             ZZ"
    @test real(expect_pauli(P"XX", bell)) ≈ 1.0 atol=1e-10
    @test real(expect_pauli(P"ZZ", bell)) ≈ 1.0 atol=1e-10
    @test abs(expect_pauli(P"X_", bell)) < 1e-10
    @test abs(expect_pauli(P"Z_", bell)) < 1e-10
    
    # Test negative phase: ⟨-Z⟩ on |0⟩ should give -1
    @test real(expect_pauli(P"-Z", s0)) ≈ -1.0 atol=1e-10
end

@testset "expect_pauli with MixedDestabilizer" begin
    bell = S"XX
             ZZ"
    md = MixedDestabilizer(bell)
    
    @test real(expect_pauli(P"XX", md)) ≈ 1.0 atol=1e-10
    @test real(expect_pauli(P"ZZ", md)) ≈ 1.0 atol=1e-10
    @test abs(expect_pauli(P"X_", md)) < 1e-10
end

@testset "is_in_stabilizer_group" begin
    s = S"XX
          ZZ"
    stb = MixedDestabilizer(s)
    
    # XX is in the group
    in_group, phase = MagicEntanglement.is_in_stabilizer_group(P"XX", stb)
    @test in_group
    @test phase ≈ 1.0
    
    # ZZ is in the group
    in_group, phase = MagicEntanglement.is_in_stabilizer_group(P"ZZ", stb)
    @test in_group
    @test phase ≈ 1.0
    
    # Identity is in the group
    in_group, phase = MagicEntanglement.is_in_stabilizer_group(P"__", stb)
    @test in_group
    @test phase ≈ 1.0
    
    # XZ is not in the group (it anticommutes with XX)
    # Actually XZ commutes with both XX and ZZ... let me check
    # XX * XZ: X anticommutes with Z at position 2, so they anticommute
    # Actually: comm(XX, XZ) - need to check carefully
    # XI is not in the group for Bell state
    in_group, _ = MagicEntanglement.is_in_stabilizer_group(P"X_", stb)
    @test !in_group
end
