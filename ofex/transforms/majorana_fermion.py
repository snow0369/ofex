from openfermion import MajoranaOperator, FermionOperator

from ofex.operators.qubit_operator_tools import dict_to_operator


# TODO : Implement the f2m mapping
def majorana_to_fermion(maj: MajoranaOperator) -> FermionOperator:
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
