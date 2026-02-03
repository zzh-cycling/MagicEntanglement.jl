module MagicEntanglement

include("Logical.jl")

export all_logical_paulis, 
       compute_logical_entropy,
       subregion_logical_paulis,
       pauli_support,
       is_supported_on,
       pauli_to_matrix,
       stabilizer_state_vector,
       expect_pauli,
       compute_logical_density_matrix,
       find_minimal_support_representative,
       is_in_stabilizer_group

end
