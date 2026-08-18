# Check: depth convergence and ensemble identity for three offset schemes.
#  fixed       — current circuit
#  offset_shot — random global brick-wall shift drawn once per trajectory
#  offset_step — random shift redrawn every time step
using QuantumClifford
using Random: MersenneTwister
using Statistics

function random_two_clifford(i, j, rng)
    return SparseGate(random_clifford(rng, 2), [i, j])
end

function full_chain_entropy(stab, L)
    return [QuantumClifford.entanglement_entropy(stab, 1:l, Val(:clip)) for l in 1:L-1]
end

function single_shot(L::Int, depth::Int, p::Float64, seed::Int, mode::Symbol)
    rng = MersenneTwister(seed)
    stab = one(Stabilizer, L)
    offset = mode === :offset_shot ? rand(rng, 0:1) : 0
    for t in 1:depth
        mode === :offset_step && (offset = rand(rng, 0:1))
        for parity in 0:1
            for i in 1:L-1
                if (i + offset) % 2 == parity
                    apply!(stab, random_two_clifford(i, i+1, rng))
                end
            end
            if (L + offset) % 2 == parity
                apply!(stab, random_two_clifford(L, 1, rng))
            end
            for i in 1:L
                rand(rng) < p && projectZ!(stab, i)
            end
        end
    end
    return full_chain_entropy(stab, L)
end

function run(L, depth, p, nshot, mode)
    Ss = zeros(L - 1)
    for shot in 1:nshot
        Ss .+= single_shot(L, depth, p, shot, mode)
    end
    return Ss ./ nshot
end

L, p = 64, 0.16
for mode in (:fixed, :offset_shot, :offset_step)
    for depth in (192, 384)
        S = run(L, depth, p, 1000, mode)
        println(rpad(string(mode), 12), " depth=$depth  S(L/2) = $(round(S[L÷2], digits=4))")
    end
end
# p=0 sanity: both ensembles must tend to the same volume law
for mode in (:fixed, :offset_step)
    S = run(32, 96, 0.0, 500, mode)
    println(rpad(string(mode), 12), " p=0, L=32: S(L/2) = $(round(S[16], digits=3)) (expect ~16)")
end
