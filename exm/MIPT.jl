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
    S_ensemble = zeros(depth+1, nshot)
    for shot in 1:nshot
        # Initial product state |0...0⟩ = |+Z⟩^⊗L
        # Each qubit is stabilized by Z_i, so we have L independent stabilizers
        stab = one(Stabilizer, L)  # Creates |0...0⟩ state with L stabilizers Z_1, Z_2, ..., Z_L
        # Layer-by-layer evolution
        S_ensemble[1, shot] = half_chain_entropy(stab, L)  # Initial entropy
        for t in 1:depth
            # ===== 1. Two-qubit Clifford gates: random even-odd alternation =====
            for i in 1:2:L-1          # Even bonds
                apply!(stab, random_two_clifford(i, i+1))
            end
            
            for i in 1:L
                if rand() < p
                    projectZ!(stab, i)   # Measure Z_i (i.e., project 0 or project 1)
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
                    projectZ!(stab, i)   # Measure Z_i (i.e., project 0 or project 1)
                end
            end
            # Calculate half-chain entropy
            S_ensemble[t+1, shot] = half_chain_entropy(stab, L)
        end
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
        # Initial product state |0...0⟩ = |+Z⟩^⊗L
        # Each qubit is stabilized by Z_i, so we have L independent stabilizers
        stab = one(Stabilizer, L)  # Creates |0...0⟩ state with L stabilizers Z_1, Z_2, ..., Z_L
        # Layer-by-layer evolution
        for t in 1:depth
            # ===== 1. Two-qubit Clifford gates: random even-odd alternation =====
            for i in 1:2:L-1          # Even bonds
                apply!(stab, random_two_clifford(i, i+1))
            end

            for i in 1:L
                if rand() < p
                    projectZ!(stab, i)   # Measure Z_i (i.e., project 0 or project 1)
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
                    projectZ!(stab, i)   # Measure Z_i (i.e., project 0 or project 1)
                end
            end
        end
        # Calculate half-chain entropy
        S_list[shot] = half_chain_entropy(stab, L)
    end
    return S_list
end

# --------- Helper: random Clifford gates ---------
# Single-qubit: Clifford(1) has 24 elements (6 without phase × 4 phases)
# Use random_clifford1(i) for Haar-random single-qubit Clifford on qubit i
random_single_clifford(i) = random_clifford1(i)

"""
    random_two_clifford(i, j)

Generate a Haar-random 2-qubit Clifford gate acting on qubits i and j.

The 2-qubit Clifford group Clifford(2) has |Clifford(2)| = 11520 elements
(720 without phases × 16 phase combinations).

For MIPT to exhibit universal behavior, we need to sample uniformly from
the full Clifford(2) group, not just a few specific gates like CNOT.

Implementation: Uses QuantumClifford.random_clifford(2) which implements
the Bravyi-Maslov algorithm for uniform sampling, wrapped in SparseGate
to apply to specific qubit indices.
"""
function random_two_clifford(i, j)
    # Generate a Haar-random 2-qubit Clifford operator
    # and wrap it in SparseGate to apply to qubits [i, j]
    return SparseGate(random_clifford(2), [i, j])
end

# --------- Half-chain von Neumann entropy ---------
function half_chain_entropy(stab::Stabilizer, L)
    A = 1:L÷2
    # QuantumClifford has built-in reduced_entropy
    QuantumClifford.entanglement_entropy(stab, A, Val(:clip))
end

# ================== Main scan ==================
println("Start scanning, depth=$depth, samples=$nshot")
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