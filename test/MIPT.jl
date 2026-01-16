using QuantumClifford
using Plots
using ProgressMeter

# ================== 参数 ==================
L      = 24             # 链长
depth  = 80             # 电路深度（足够饱和）
nshot  = 60             # 每个 p 点的随机电路样本数
p_list = 0.0:0.05:1.0   # 测量概率扫描
# =========================================

"""
    run_mipt_clifford(L, depth, p, nshot)

经典 MIPT 的 Clifford 模拟。
返回 (p, S_list)，S_list 长度为 nshot，存半链熵。
"""
function run_mipt_clifford(L, depth, p, nshot)
    S_list = zeros(nshot)
    for shot in 1:nshot
        # 初始乘积态 |0⟩^L
        stab = zero(Stabilizer, L)
        # 逐层演化
        for t in 1:depth
            # ===== 1. 双比特 Clifford 门：随机偶奇交错 =====
            for i in 1:2:L-1          # 偶链接
                apply!(stab, random_two_clifford(i, i+1))
            end
            for i in 2:2:L-1          # 奇链接
                apply!(stab, random_two_clifford(i, i+1))
            end
            # ===== 2. 单比特随机 Clifford =====
            for i in 1:L
                apply!(stab, random_single_clifford(i))
            end
            # ===== 3. 以概率 p 做 Z 测量 =====
            for i in 1:L
                if rand() < p
                    projectX!(stab, i)   # 测量 Z_i（即 project+ 或 project-）
                end
            end
        end
        # 计算半链熵
        S_list[shot] = half_chain_entropy(stab, L)
    end
    return S_list
end

# --------- 辅助：随机 Clifford 门 ---------
random_single_clifford(i) = rand([sHadamard(i), sPhase(i), sX(i), sY(i), sZ(i)])

function random_two_clifford(i, j)
    # 从常见双比特 Clifford 池子里随机挑一个
    gates = [
        sCNOT(i,j), sCNOT(j,i)
        # , sCZ(i,j),sSWAP(i,j), sCY(i,j), sCX(j,i)
    ]
    rand(gates)
end

# --------- 半链 von Neumann 熵 ---------
function half_chain_entropy(stab::Stabilizer, L)
    A = 1:L÷2
    # QuantumClifford 内置 reduced_entropy
    QuantumClifford.entanglement_entropy(stab, A)
end

# ================== 主扫描 ==================
println("开始扫描，L=$L, depth=$depth, 样本=$nshot")
S_mean = Float64[]
S_err  = Float64[]
@showprogress for p in p_list
    Ss = run_mipt_clifford(L, depth, p, nshot)
    push!(S_mean, mean(Ss))
    push!(S_err,  std(Ss)/sqrt(nshot))
end

# ================== 画图 ==================
plot(p_list, S_mean, yerr=S_err,
     linewidth=2, marker=:circle, markersize=4,
     xlabel="测量概率 p", ylabel="半链 von Neumann 熵 S(L/2)",
     title="Clifford MIPT, L=$L, depth=$depth",
     legend=false, grid=true)
savefig("mipt_clifford_L$L.png")
println("结果已保存为 mipt_clifford_L$L.png")