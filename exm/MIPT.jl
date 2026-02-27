using QuantumClifford
using ProgressMeter
using Statistics
using JLD2
using Random: MersenneTwister
using Distributed

# Add worker processes if not already added
if nprocs() == 1
    addprocs()  # Add worker processes, to be adjust as needed
end

@everywhere begin
    using QuantumClifford
    using Random: MersenneTwister
    using Statistics
    
    # --------- Half-chain von Neumann entropy ---------
    function half_chain_entropy(stab::Stabilizer, L::Int)
        A = 1:L÷2
        QuantumClifford.entanglement_entropy(stab, A, Val(:clip))
    end
    
    # --------- full-chain von Neumann entropy ---------
    function full_chain_entropy(stab::Stabilizer, L::Int)
        A = [1:i for i in 1:L-1]
        QuantumClifford.entanglement_entropy.(Ref(stab), A, Val(:clip))
    end

    # --------- Helper: random Clifford gates ---------
    """
        random_two_clifford(i, j, rng)
    
    Generate a Haar-random 2-qubit Clifford gate acting on qubits i and j.
    """
    function random_two_clifford(i, j, rng::MersenneTwister)
        return SparseGate(random_clifford(rng, 2), [i, j])
    end
    
    """
        single_shot_dynamics(L, depth, p, seed) -> Vector{Float64}
    
    Run a single trajectory of the MIPT dynamics.
    Returns entropy at each time step (length depth+1).
    """
    function single_shot_dynamics(L::Int, depth::Int, p::Float64, seed::Int=0, pbc::Bool=true)
        try 
            rng = MersenneTwister(seed)
            S_trajectory = zeros(depth + 1)
            Sscaling_trajectory = zeros(depth + 1, L - 1)
    
            stab = one(Stabilizer, L)
            S_trajectory[1] = half_chain_entropy(stab, L)
            
            for t in 1:depth
                # Odd bonds
                for i in 1:2:L-1
                    apply!(stab, random_two_clifford(i, i+1, rng))
                end

                for i in 1:L
                    if rand(rng) < p
                        projectZ!(stab, i)
                    end
                end
        
                # Even bonds
                for i in 2:2:L-1
                    apply!(stab, random_two_clifford(i, i+1, rng))
                end

                if pbc # Assue L is even
                    apply!(stab, random_two_clifford(L, 1, rng))
                end
                
                for i in 1:L
                    if rand(rng) < p
                        projectZ!(stab, i)
                    end
                end
                
                S_trajectory[t+1] = half_chain_entropy(stab, L)
                Sscaling_trajectory[t+1, :] = full_chain_entropy(stab, L)
            end
            
            return (p, seed, S_trajectory, Sscaling_trajectory, :success, nothing)
        catch e 
            return (p, seed, zeros(depth + 1), zeros(depth + 1, L - 1), :failure, e)
        end
    end
    
    """
        single_shot_mipt(L, depth, p, seed) -> Float64
    
    Run a single trajectory and return the final half-chain entropy.
    """
    function single_shot_mipt(L::Int, depth::Int, p::Float64, seed::Int=0, pbc::Bool=true)
        try 
            rng = MersenneTwister(seed)
            stab = one(Stabilizer, L)
            
            for t in 1:depth
                # Odd bonds
                for i in 1:2:L-1
                    apply!(stab, random_two_clifford(i, i+1, rng))
                end
        
                for i in 1:L
                    if rand(rng) < p
                        projectZ!(stab, i)
                    end
                end
        
                # Even bonds
                for i in 2:2:L-1
                    apply!(stab, random_two_clifford(i, i+1, rng))
                end
                
                if pbc # Assue L is even
                    apply!(stab, random_two_clifford(L, 1, rng))
                end

                for i in 1:L
                    if rand(rng) < p
                        projectZ!(stab, i)
                    end
                end
            end
            
            return (p, seed, half_chain_entropy(stab, L), full_chain_entropy(stab, L), :success, nothing)
        catch e 
            return (p, seed, 0.0, zeros(L-1), :failure, e)
        end
    end
end


"""
    clifford_dynamics(L, depth, p, nshot) -> Matrix{Float64}

Run nshot trajectories of MIPT dynamics using distributed parallel map.
Returns S_ensemble of size (depth+1, nshot).
"""
function clifford_dynamics(L::Int, depth::Int, p::Float64, nshot::Int)
    # Use pmap to distribute work across processes
    results = pmap(shot -> single_shot_dynamics(L, depth, p, shot), 1:nshot)
    S_trajectories = [res[3] for res in results if res[5] == :success]
    S_scaling_trajectories = [res[4] for res in results if res[5] == :success]
    # Convert vector of vectors to matrix
    S_ensemble = hcat(S_trajectories...)
    S_scaling_ensemble = hcat(S_scaling_trajectories...)
    
    failed_tasks = [(p, seed, error)
                        for (p, seed, _, _, status, error) in results 
                        if status != :success]
        
    success_count = count(r -> r[3] == :success, results)
    failed_count = length(failed_tasks)
    
    println("Successes: $success_count")
    println("Failures: $failed_count")
    println("Failed tasks errors: $failed_tasks")
    return S_ensemble, S_scaling_ensemble
end

"""
    run_mipt_clifford(L, depth, p, nshot)

Distributed parallel Clifford simulation of MIPT using pmap.
Returns S_list of length nshot containing final half-chain entropies.
"""
function run_mipt_clifford(L::Int, depth::Int, p::Float64, nshot::Int)
    # Use pmap to distribute work across processes
    results = pmap(shot -> single_shot_mipt(L, depth, p, shot), 1:nshot)
    S_list = [res[3] for res in results if res[5] == :success]
    S_scaling_list = [res[4] for res in results if res[5] == :success]

    failed_tasks = [(p, seed, error)
                        for (p, seed, _, _, status, error) in results 
                        if status != :success]
    success_count = count(r -> r[3] == :success, results)
    failed_count = length(failed_tasks)

    println("Successes: $success_count")
    println("Failures: $failed_count")  
    println("Failed tasks errors: $failed_tasks")
    return S_list, S_scaling_list
end

function obtain_depth(L::Int)
     # Circuit depth (enough to saturate), at least L/2
    return max(50, Int(0.5L))
end
# ================== Parallel Main Scan ==================
"""
    run_parameter_scan(Llist, p_list, depth, nshot; show_progress=true)

Run the full parameter scan with multithreading over shots.
Returns (S_mean, S_err) matrices.
"""
function run_parameter_scan(Llist, p_list, nshot::Int; show_progress=true)
    np = length(p_list)
    nL = length(Llist)
    S_mean = zeros(np, nL)
    S_err  = zeros(np, nL)
    
    # Create list of all (ip, iL, p, L) tasks
    tasks = [(ip, iL, p, L) for (ip, p) in enumerate(p_list) for (iL, L) in enumerate(Llist)]

    println("Total tasks: $(length(tasks))")
  
    if show_progress
        prog = Progress(length(tasks); desc="MIPT scan: ")
    end
    
    # Each task runs multithreaded shots internally
    for (ip, iL, p, L) in tasks
        Ss, S_scaling = run_mipt_clifford(L, obtain_depth(L), Float64(p), nshot)
        save(joinpath("exm/data/MIPT/L$(L)", "L$(L)_p$(round(p, digits=3))_scaling.jld2"), 
        "S_scaling", mean(hcat(S_scaling...), dims=2))
        S_mean[ip, iL] = mean(Ss)
        S_err[ip, iL]  = std(Ss) / sqrt(nshot)
        show_progress && next!(prog)
    end
    # summary report
    println("\n=== Processing Complete ===")
    return S_mean, S_err
end

# ================== Parameters ==================

Llist  = 2 .^collect(2:8)             # Chain length
nshot  = 5000            # Number of random circuit samples per p point
p_list = 0.0:0.02:0.3   # Measurement probability scan
# ================================================

# Print process info
println("Running with $(nprocs()) processes ($(nworkers()) workers)")
println("To add more workers: using Distributed; addprocs(N)")

println("Start scanning, samples=$nshot")
S_mean, S_err = run_parameter_scan(Llist, p_list, nshot)
savepath = "exm/data/MIPT/"
mkpath(savepath)
save(joinpath(savepath, "MIPT_L$(Llist[end]).jld2"), 
"S_mean", S_mean, 
"S_err", S_err) 