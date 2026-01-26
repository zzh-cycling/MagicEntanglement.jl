using QuantumClifford
using Plots
using ProgressMeter

# ================== Parameters ==================
L      = 24             # Chain length
depth  = 80             # Circuit depth (enough to saturate)
nshot  = 60             # Number of random circuit samples per p point
p_list = 0.0:0.05:1.0   # Measurement probability scan
# ================================================

"""
    run_mipt_clifford(L, depth, p, nshot)

Classical Clifford simulation of MIPT.
Returns (p, S_list), where S_list has length nshot and stores half-chain entropies.
"""
function run_mipt_clifford(L, depth, p, nshot)
    S_list = zeros(nshot)
    for shot in 1:nshot
        # Initial product state |0⟩^L
        stab = zero(Stabilizer, L)
        # Layer-by-layer evolution
        for t in 1:depth
            # ===== 1. Two-qubit Clifford gates: random even-odd alternation =====
            for i in 1:2:L-1          # Even bonds
                apply!(stab, random_two_clifford(i, i+1))
            end
            for i in 2:2:L-1          # Odd bonds
                apply!(stab, random_two_clifford(i, i+1))
            end
            # ===== 2. Single-qubit random Clifford gates =====
            for i in 1:L
                apply!(stab, random_single_clifford(i))
            end
            # ===== 3. Z measurement with probability p =====
            for i in 1:L
                if rand() < p
                    projectX!(stab, i)   # Measure Z_i (i.e., project+ or project-)
                end
            end
        end
        # Calculate half-chain entropy
        S_list[shot] = half_chain_entropy(stab, L)
    end
    return S_list
end

# --------- Helper: random Clifford gates ---------
random_single_clifford(i) = rand([sHadamard(i), sPhase(i), sX(i), sY(i), sZ(i)])

function random_two_clifford(i, j)
    # Randomly pick from a common pool of two-qubit Clifford gates
    gates = [
        sCNOT(i,j), sCNOT(j,i)
        # , sCZ(i,j),sSWAP(i,j), sCY(i,j), sCX(j,i)
    ]
    rand(gates)
end

# --------- Half-chain von Neumann entropy ---------
function half_chain_entropy(stab::Stabilizer, L)
    A = 1:L÷2
    # QuantumClifford has built-in reduced_entropy
    QuantumClifford.entanglement_entropy(stab, A)
end

# ================== Main scan ==================
println("Start scanning, L=$L, depth=$depth, samples=$nshot")
S_mean = Float64[]
S_err  = Float64[]
@showprogress for p in p_list
    Ss = run_mipt_clifford(L, depth, p, nshot)
    push!(S_mean, mean(Ss))
    push!(S_err,  std(Ss)/sqrt(nshot))
end

# ================== Plotting ==================
plot(p_list, S_mean, yerr=S_err,
    linewidth=2, marker=:circle, markersize=4,
    xlabel="Measurement probability p", ylabel="Half-chain von Neumann entropy S(L/2)",
    title="Clifford MIPT, L=$L, depth=$depth",
    legend=false, grid=true)
savefig("mipt_clifford_L$L.png")
println("Result saved as mipt_clifford_L$L.png")