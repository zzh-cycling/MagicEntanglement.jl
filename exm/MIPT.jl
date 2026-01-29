using QuantumClifford
using Plots
using ProgressMeter
using Statistics
using Plots
using JLD2
using LaTeXStrings

# ================== Parameters ==================

Llist  = collect(20:10:80)             # Chain length
depth  = 80             # Circuit depth (enough to saturate)
nshot  = 1000            # Number of random circuit samples per p point
p_list = 0.0:0.02:0.3   # Measurement probability scan
# ================================================

function clifford_dynamics(L, depth, p, nshot)
    S_ensemble = zeros(depth, nshot)
    for shot in 1:nshot
        # Initial product state |0⟩^L
        stab =  Stabilizer([PauliOperator([zeros(Bool, L);ones(Bool, L)])]) # All +1 eigenstate of Z_i
        # Layer-by-layer evolution
        for t in 1:depth
            # ===== 1. Two-qubit Clifford gates: random even-odd alternation =====
            for i in 1:2:L-1          # Even bonds
                apply!(stab, random_two_clifford(i, i+1))
            end
            
            for i in 1:L
                if rand() < p
                    projectZ!(stab, i)   # Measure Z_i (i.e., project+ or project-)
                end
            end

            for i in 2:2:L-1          # Odd bonds
                apply!(stab, random_two_clifford(i, i+1))
            end
            # ===== 2. Single-qubit random Clifford gates =====
            # for i in 1:L
            #     apply!(stab, random_single_clifford(i))
            # end
            # ===== 3. Z measurement with probability p =====
            for i in 1:L
                if rand() < p
                    projectZ!(stab, i)   # Measure Z_i (i.e., project+ or project-)
                end
            end
        end
        # Calculate half-chain entropy
        S_ensemble[t, shot] = half_chain_entropy(stab, L)
    end
    return S_ensemble
end

"""
    run_mipt_clifford(L, depth, p, nshot)

Classical Clifford simulation of MIPT.
Returns (p, S_list), where S_list has length nshot and stores half-chain entropies.
"""
function run_mipt_clifford(L, depth, p, nshot)
    S_list = zeros(nshot)
    for shot in 1:nshot
        # Initial product state |0⟩^L
        stab =  Stabilizer([PauliOperator([zeros(Bool, L);ones(Bool, L)])]) # All +1 eigenstate of Z_i
        # Layer-by-layer evolution
        for t in 1:depth
            # ===== 1. Two-qubit Clifford gates: random even-odd alternation =====
            for i in 1:2:L-1          # Even bonds
                apply!(stab, random_two_clifford(i, i+1))
            end

            for i in 1:L
                if rand() < p
                    projectZ!(stab, i)   # Measure Z_i (i.e., project+ or project-)
                end
            end

            for i in 2:2:L-1          # Odd bonds
                apply!(stab, random_two_clifford(i, i+1))
            end
            # ===== 2. Single-qubit random Clifford gates =====
            # for i in 1:L
            #     apply!(stab, random_single_clifford(i))
            # end
            # ===== 3. Z measurement with probability p =====
            for i in 1:L
                if rand() < p
                    projectZ!(stab, i)   # Measure Z_i (i.e., project+ or project-)
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
    QuantumClifford.entanglement_entropy(stab, A, Val(:clip))
end

# ================== Main scan ==================
println("Start scanning, L=$L, depth=$depth, samples=$nshot")
S_mean = zeros(length(p_list), length(Llist))
S_err  = zeros(length(p_list), length(Llist))
@showprogress for (ip, p) in enumerate(p_list)
    for (iL, L) in enumerate(Llist)
        Ss = run_mipt_clifford(L, depth, p, nshot)
        S_mean[ip, iL] = mean(Ss)
        S_err[ip, iL]  = std(Ss)/sqrt(nshot)
    end
end

# ================== Plotting ==================
plot(Llist, S_mean', yerr=S_err', label=p_list',
line_z =p_list', color=:blues, marker_z = p_list',
linewidth=2, marker=:circle, markersize=4,
    xlabel=L"L", ylabel=L"S(L/2)", grid=true, 
    legend_background_color=nothing, legend_foreground_color=nothing, colorbar=false)
savefig("mipt_clifford.pdf")
println("Result saved as mipt_clifford.png")