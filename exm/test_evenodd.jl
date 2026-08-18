# Experiment: origin of the even-odd staggering of S(l) in the Clifford MIPT.
#
# Three circuit ensembles, all brick-wall random two-qubit Clifford + Z measurements:
#   :fixed   — current implementation (odd layer, measure, even layer + wrap bond, measure)
#   :singleq — :fixed + a Haar-random single-qubit Clifford on every site after each layer
#   :offset  — :fixed but with the brick-wall matching shifted by one site randomly
#              every time step (restores one-site translation invariance of the ensemble)
#
# The staggering is quantified by regressing S(l) on [1, log-chord, (-1)^l] in the bulk.

using QuantumClifford
using Random: MersenneTwister
using Statistics
using LinearAlgebra: mul!

function random_two_clifford(i, j, rng)
    return SparseGate(random_clifford(rng, 2), [i, j])
end

function full_chain_entropy(stab, L)
    return [QuantumClifford.entanglement_entropy(stab, 1:l, Val(:clip)) for l in 1:L-1]
end

function single_shot(L::Int, depth::Int, p::Float64, seed::Int, mode::Symbol)
    rng = MersenneTwister(seed)
    stab = one(Stabilizer, L)
    for t in 1:depth
        offset = mode === :offset ? rand(rng, 0:1) : 0
        # two matchings of the ring, swapped when offset == 1
        for parity in 0:1
            for i in 1:L-1
                if (i + offset) % 2 == parity
                    apply!(stab, random_two_clifford(i, i+1, rng))
                end
            end
            if (L + offset) % 2 == parity
                apply!(stab, random_two_clifford(L, 1, rng))
            end
            if mode === :singleq
                for i in 1:L
                    apply!(stab, SparseGate(random_clifford(rng, 1), [i]))
                end
            end
            for i in 1:L
                rand(rng) < p && projectZ!(stab, i)
            end
        end
    end
    return full_chain_entropy(stab, L)
end

function ensemble_mean_S(L, depth, p, nshot, mode)
    Ss = zeros(L - 1)
    for shot in 1:nshot
        Ss .+= single_shot(L, depth, p, shot, mode)
    end
    return Ss ./ nshot
end

# stagger coefficient from regression S(l) ~ a + b*logChord + c*(-1)^l on the bulk
function stagger_fit(S, L)
    logChord(l) = log(sin(pi * l / L)) / 3
    bulk = (L÷4):(3L÷4)
    X = hcat(ones(length(bulk)), logChord.(bulk), [Float64((-1)^l) for l in bulk])
    coef = X \ S[bulk]
    return coef[3]  # amplitude of (-1)^l
end

L, depth, p, nshot = 64, 192, 0.16, 2000
for mode in (:fixed, :singleq, :offset)
    S = ensemble_mean_S(L, depth, p, nshot, mode)
    amp = stagger_fit(S, L)
    even_mean = mean(S[2:2:L-2]); odd_mean = mean(S[3:2:L-3])
    println(rpad(string(mode), 8),
            "  stagger amplitude = $(round(amp, digits=4))",
            "  <S_even>-<S_odd> = $(round(even_mean - odd_mean, digits=4))",
            "  S(L/2) = $(round(S[L÷2], digits=4))")
end
