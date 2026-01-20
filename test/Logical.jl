using MagicEntanglement
using QuantumClifford   

@testset "all_logical_paulis" begin
    s = S"XXXX+ZIZI+IZIZ" # [[4, 1, 2]] code
    stb = MixedDestabilizer(s)
    logicals = all_logical_paulis(stb)
    @test length(logicals) == 16  # 2 logical qubits -> 4^2 = 16 logical Paulis
    @test logicals == unique(logicals)  # no duplicates
end