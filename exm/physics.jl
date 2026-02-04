using QuantumClifford

s = S"XXXX+ZIZI+IZIZ"
canonicalize!(s)

m = stab_to_gf2(s)
QuantumClifford.gf2_nullspace(Int.(m))

s1 = S"XXXX
       ZZII
       IZZI
       IIZZ"
matrix = stab_to_gf2(s1)

kernel_ops = normalizer(tab(s))
length(kernel_ops)

using QuantumClifford

# 创建一个量子纠错码的 stabilizer
steane_code = S"XXXXXXX
                 ZZZZIII
                 IZZZZII
                 IIZZZZI
                 IIIZZZZ"

# 转换为 GF(2) 矩阵
check_matrix = stab_to_gf2(steane_code)
lg = MixedDestabilizer(steane_code)
length(lg)

# 计算 kernel (nullspace)
kernel = QuantumClifford.gf2_nullspace(Int.(check_matrix))

# 或者使用 normalizer
logical_ops = normalizer(tab(steane_code))



