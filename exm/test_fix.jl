# Verification of the patched exm/MIPT.jl single-shot dynamics.
# The circuit body below is a verbatim copy of the patched single_shot_mipt
# loop in exm/MIPT.jl (per-shot random brick-wall offset).
using QuantumClifford
using Random: MersenneTwister
using Statistics

function random_two_clifford(i, j, rng)
    return SparseGate(random_clifford(rng, 2), [i, j])
end

function full_chain_entropy(stab, L)
    return [QuantumClifford.entanglement_entropy(stab, 1:l, Val(:clip)) for l in 1:L-1]
end

function single_shot_patched(L::Int, depth::Int, p::Float64, seed::Int, pbc::Bool=true)
    rng = MersenneTwister(seed)
    stab = one(Stabilizer, L)

    # Per-shot brick-wall offset, see single_shot_dynamics above.
    offset = rand(rng, 0:1)
    nbonds = pbc ? L : L - 1

    for t in 1:depth
        # Gate layer: bonds (i, i+1 mod L) with (i + offset) odd
        for i in 1:nbonds
            if (i + offset) % 2 == 1
                apply!(stab, random_two_clifford(i, mod(i, L) + 1, rng))
            end
        end

        for i in 1:L
            if rand(rng) < p
                projectZ!(stab, i)
            end
        end

        # Gate layer: bonds (i, i+1 mod L) with (i + offset) even
        for i in 1:nbonds
            if (i + offset) % 2 == 0
                apply!(stab, random_two_clifford(i, mod(i, L) + 1, rng))
            end
        end

        # (the wrap bond (L, 1) is included via nbonds when pbc)

        for i in 1:L
            if rand(rng) < p
                projectZ!(stab, i)
            end
        end
    end

    return full_chain_entropy(stab, L)
end

function stagger_fit(S, L)
    logChord(l) = log(sin(pi * l / L)) / 3
    bulk = (L÷4):(3L÷4)
    X = hcat(ones(length(bulk)), logChord.(bulk), [Float64((-1)^l) for l in bulk])
    return (X \ S[bulk])[3]
end

L, depth, p, nshot = 64, 192, 0.16, 2000
S = zeros(L - 1)
for shot in 1:nshot
    S .+= single_shot_patched(L, depth, p, shot)
end
S ./= nshot
println("patched: stagger amplitude = $(round(stagger_fit(S, L), digits=4))  (fixed circuit was -0.117)")
println("patched: S(L/2) = $(round(S[L÷2], digits=4))  (fixed circuit was ~4.08)")
