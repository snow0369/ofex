from openfermion import MajoranaOperator, FermionOperator

from ofex.operators.symbolic_operator_tools import dict_to_operator

__all__ = ["majorana_to_fermion"]

# TODO : Implement the f2m mapping
def majorana_to_fermion(maj: MajoranaOperator) -> FermionOperator:
    """
    Convert a MajoranaOperator into its equivalent FermionOperator representation 
    using the following mapping rules:

    ∙ m_{2k} = a_k† + a_k
    ∙ m_{2k+1} = 1j * (a_k† - a_k)
    
    Each Majorana operator is expressed in terms of Fermionic creation (a_k†) 
    and annihilation (a_k) operators according to these definitions.
    
    Args:
        maj (MajoranaOperator): A MajoranaOperator instance to be converted into
            its equivalent FermionOperator representation.
    
    Returns:
        FermionOperator: The resulting FermionOperator equivalent to the input 
            MajoranaOperator after applying the mapping rules.
    """
    ret_f_sum = FermionOperator()
    for m_op, coeff in maj.terms.items():
        f_sum = FermionOperator.identity() * coeff
        for k in m_op:
            if k % 2 == 0:
                f_sum = f_sum * dict_to_operator({
                    ((k // 2, 1),): 1.0,
                    ((k // 2, 0),): 1.0
                }, FermionOperator)
            else:
                f_sum = f_sum * dict_to_operator({
                    ((k // 2, 1),): 1j,
                    ((k // 2, 0),): -1j
                }, FermionOperator)
        ret_f_sum = ret_f_sum + f_sum
    return ret_f_sum
