using QuantumClifford
using CairoMakie
using ProgressMeter
using Statistics
using JLD2
using Random: MersenneTwister
using Plots: plot, savefig
using LaTeXStrings
using Distributed

# Add worker processes if not already added
if nprocs() == 1
    addprocs(8)  # Add 8 worker processes, adjust as needed
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
    function single_shot_dynamics(L::Int, depth::Int, p::Float64, seed::Int=0)
        try 
            rng = MersenneTwister(seed)
            S_trajectory = zeros(depth + 1)
            Sscaling_trajectory = zeros(depth + 1, L - 1)
    
            stab = one(Stabilizer, L)
            S_trajectory[1] = half_chain_entropy(stab, L)
            
            for t in 1:depth
                # Even bonds
                for i in 1:2:L-1
                    apply!(stab, random_two_clifford(i, i+1, rng))
                end
                
                for i in 1:L
                    if rand(rng) < p
                        projectZ!(stab, i)
                    end
                end
        
                # Odd bonds
                for i in 2:2:L-1
                    apply!(stab, random_two_clifford(i, i+1, rng))
                end
                
                for i in 1:L
                    if rand(rng) < p
                        projectZ!(stab, i)
                    end
                end
                
                S_trajectory[t+1] = half_chain_entropy(stab, L)
                Sscaling_trajectory[t+1, :] = full_chain_entropy(stab, L)
            end
            
            return S_trajectory, Sscaling_trajectory, :success
        catch e 
            return p, seed, :failure, e
        end
    end
    
    """
        single_shot_mipt(L, depth, p, seed) -> Float64
    
    Run a single trajectory and return the final half-chain entropy.
    """
    function single_shot_mipt(L::Int, depth::Int, p::Float64, seed::Int=0)
        try 
            rng = MersenneTwister(seed)
            stab = one(Stabilizer, L)
            
            for t in 1:depth
                # Even bonds
                for i in 1:2:L-1
                    apply!(stab, random_two_clifford(i, i+1, rng))
                end
        
                for i in 1:L
                    if rand(rng) < p
                        projectZ!(stab, i)
                    end
                end
        
                # Odd bonds
                for i in 2:2:L-1
                    apply!(stab, random_two_clifford(i, i+1, rng))
                end
                
                for i in 1:L
                    if rand(rng) < p
                        projectZ!(stab, i)
                    end
                end
            end
            
            return half_chain_entropy(stab, L), full_chain_entropy(stab, L), :success
        catch e 
            return p, seed, :failure, e
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
    S_trajectories = [res[1] for res in results if res[3] == :success]
    S_scaling_trajectories = [res[2] for res in results if res[3] == :success]
    # Convert vector of vectors to matrix
    S_ensemble = hcat(S_trajectories...)
    S_scaling_ensemble = hcat(S_scaling_trajectories...)
    
    failed_tasks = [(p, seed, error)
                        for (p, seed, status, error) in results 
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
    S_list = [res[1] for res in results if res[3] == :success]
    S_scaling_list = [res[2] for res in results if res[3] == :success]

    failed_tasks = [(p, seed, error) 
                        for (p, seed, status, error) in results 
                        if status != :success]
    success_count = count(r -> r[3] == :success, results)
    failed_count = length(failed_tasks)

    println("Successes: $success_count")
    println("Failures: $failed_count")  
    println("Failed tasks errors: $failed_tasks")
    return S_list, S_scaling_list
end

# ================== Parallel Main Scan ==================
"""
    run_parameter_scan(Llist, p_list, depth, nshot; show_progress=true)

Run the full parameter scan with multithreading over shots.
Returns (S_mean, S_err) matrices.
"""
function run_parameter_scan(Llist, p_list, depth::Int, nshot::Int; show_progress=true)
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
        Ss, Ss_scaling = run_mipt_clifford(L, depth, Float64(p), nshot)
        S_mean[ip, iL] = mean(Ss)
        S_err[ip, iL]  = std(Ss) / sqrt(nshot)
        show_progress && next!(prog)
    end
    # summary report
    println("\n=== Processing Complete ===")
    return S_mean, S_err
end

# ================== Parameters ==================

Llist  = collect(10:4:50)             # Chain length
depth  = 80             # Circuit depth (enough to saturate)
nshot  = 1000            # Number of random circuit samples per p point
p_list = 0.0:0.02:0.3   # Measurement probability scan
# ================================================

# Print process info
println("Running with $(nprocs()) processes ($(nworkers()) workers)")
println("To add more workers: using Distributed; addprocs(N)")

println("Start scanning, depth=$depth, samples=$nshot")
S_mean, S_err = run_parameter_scan(Llist, p_list, depth, nshot)
savepath = "exm/data/MIPT/"
mkpath(savepath)
save(joinpath(savepath, "data.jld2"), 
"S_mean", S_mean, 
"S_err", S_err) 

S_mean, S_err = load(joinpath(savepath, "data.jld2"), "S_mean", "S_err")
# ================== Plotting ==================

fig = plot(Llist, S_mean', yerr=S_err', label=p_list',
line_z =p_list', color=:blues, marker_z = p_list',
linewidth=2, marker=:circle, markersize=4,
    xlabel=L"L", ylabel=L"S(L/2)", grid=true, 
    legend_background_color=nothing, legend_foreground_color=nothing, colorbar=false)
savefig("mipt_clifford.pdf")
println("Result saved as mipt_clifford.pdf")