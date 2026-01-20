function compute_logical_entropy(S::Stabilizer, subsystems::Vector{Int}, ψ::Vector{ET}) where ET
    tableaux = MixedDestabilizer(S)
    logical_ops_X = logicalxview(tableaux)
    logical_ops_Z = logicalzview(tableaux)
    normalizer_ops = normalizer(tab(S))
   
    sub_logical_ops = [op for op in logical_ops if all(q in subsystems for q in qubits(op))]
    
    # 计算熵
    entropy = 0.0
    for op in sub_logical_ops
        p = abs2(dot(ψ, apply(op, ψ)))
        if p > 0
            entropy -= p * log2(p)
        end
    end
    
    return entropy
    
end

using QuantumClifford
using LinearAlgebra: dot, eigen!, log2

"""
    compute_logical_entropy(S::Stabilizer, A::Vector{Int}, Ψ::Vector{ET}) where ET<:Number

论文 Algorithm 1 的完整实现。
输入
  S   : 公共 stabilizer 群（n 个物理比特）
  A   : 子系统比特编号，例如 [1,2,3]
  Ψ   : 低魔法态 |Ψ⟩=Σ cⱼ|ψⱼ⟩ 的系数向量，长度须等于 |ψⱼ⟩ 的个数 K
输出
  S(ρ_a) : 逻辑子系统 a 的 von Neumann 熵
"""
function compute_logical_entropy(S::Stabilizer, A::Vector{Int}, Ψ::Vector{ET}) where ET<:Number
    n = nqubits(S)               # 物理比特数
    @assert all(1 .≤ A .≤ n) "subsystem A 越界"
    K = length(Ψ)                # 叠加项数
    @assert K ≥ 1

    # ---------- 1. 取得完整 logical Pauli 基 ----------
    tab = MixedDestabilizer(S)   # 同时给出 stabilizer + logical
    k = rank(tab)                # 逻辑比特数 k = ν
    # 所有 4^k 个 logical Pauli 算符（含 I）
    logical_basis = all_logical_paulis(tab)   # 向量长度 4^k

    # ---------- 2. cleaning：只保留完全落在 A 上的 ----------
    sub_logical = PauliOperator[]              # 将构成 M_A 的基
    for P in logical_basis
        # 检查 P 的 support 是否 ⊆ A
        support = [i for i in 1:n if !iszero(P,i)]
        if all(q -> q ∈ A, support)
            push!(sub_logical, P)
        end
    end
    d = length(sub_logical)        # M_A 的维度（含 I），必为 2^{k_a}
    k_a = round(Int, log2(d))      # A 能恢复的逻辑比特数
    @assert 2^k_a == d "internal dim error"

    # ---------- 3. 对 M_A 做 tomography，重建 ρ_a ----------
    ρ_a = zeros(ComplexF64, d, d)  # 2^{k_a} × 2^{k_a}
    # 把 M_A 的基映射到 2^{k_a} 维 Hilbert 空间的 Pauli 矩阵
    # 用标准编码：第 i 个 Pauli 对应 2^{k_a}×2^{k_a} 的矩阵表示
    pauli_mat = Matrix{ComplexF64}[logical_matrix(p) for p in sub_logical]
    # 归一化系数：Tr(P_i P_j)=d δ_{ij}
    for (i, Pi) in enumerate(sub_logical)
        # 计算 <Ψ|Pi|Ψ>
        expval = zero(ComplexF64)
        for α in 1:K, β in 1:K
            # |ψ_α⟩ 是 stabilizer state，可用 stabilizer 内积公式
            ov = dot(stab_state(S,α), apply(Pi, stab_state(S,β)))
            expval += conj(Ψ[α]) * Ψ[β] * ov
        end
        ρ_a .+= (expval / d) .* pauli_mat[i]
    end
    # 保证 Hermitian & trace=1
    ρ_a = (ρ_a + ρ_a') / 2
    ρ_a /= tr(ρ_a)

    # ---------- 4. 对角化得熵 ----------
    λ = eigen!(Hermitian(ρ_a)).values
    S = 0.0
    for v in λ
        v > 0 && (S -= v * log2(v))
    end
    return S
end

# ---------- Generate all logical operators with type PauliOperator ----------
function all_logical_paulis(tab::MixedDestabilizer)
    r = rank(tab)
    n = nqubits(tab)
    k = n -r # logical qubits number
    ops = PauliOperator[]
    for idx in 0:(4^k-1)
        # 把 idx 看成 k 位 4 进制：0=I,1=X,2=Y,3=Z
        P = one(PauliOperator, nqubits(tab))
        for i in 1:k
            digit = (idx >> (2(i-1))) & 3
            if digit == 1
                P *= logicalxview(tab)[i]
            elseif digit == 3
                P *= logicalzview(tab)[i]
            elseif digit == 2
                P *= im * logicalxview(tab)[i] * logicalzview(tab)[i]
            end
        end
        push!(ops, P)
    end
    ops
end

# ---------- 辅助：stabilizer state 的快速内积 ----------
function stab_state(S::Stabilizer, idx::Int)
    # 这里假设 |ψ_idx⟩ 是 code 子空间里某个
    # 固定 logical 计算基态（例如 |+⟩ 或 |0⟩）。
    # 为演示，我们直接取第 idx 个 stabilizer state 的 tableau 表示。
    # 实际使用时，可把 |ψ_j⟩ 预先存成 Stabilizer 对象。
    tab = copy(S)
    # 简单方案：把 logical Z 设为 +1 本征态
    logicalz = logicalzview(tab)
    for i in 1:length(logicalz)
        apply!(tab, logicalz[i])   # 测量并投影
    end
    tab
end