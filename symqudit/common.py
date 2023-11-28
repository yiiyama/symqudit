from collections import defaultdict
from sympy import Add, I, KroneckerDelta, Mul, Order, Rational, S, sqrt, sympify
from sympy.physics.quantum import BraBase, InnerProduct, KetBase, OuterProduct, State, StateBase


class DiscreteState(State, StateBase):
    pass

class DiscreteKet(DiscreteState, KetBase):
    @classmethod
    def dual_class(self):
        return DiscreteBra

    def __mul__(self, other):
        c, nc = other.args_cnc()
        if len(nc) == 1 and isinstance(nc[0], DiscreteBra):
            return Mul(*c) * DiscreteOuterProduct(self, nc[0])

        return super().__mul__(other)

    def _eval_innerproduct_DiscreteBra(self, bra, **hints):
        if len(self.args) != len(bra.args):
            raise ValueError('Cannot multiply a ket that has a different number of labels.')

        result = S.One
        for arg, bra_arg in zip(self.args, bra.args):
            result *= KroneckerDelta(arg, bra_arg)
        return result

class DiscreteBra(DiscreteState, BraBase):
    @classmethod
    def dual_class(self):
        return DiscreteKet

    def __mul__(self, other):
        c, nc = other.args_cnc()
        if len(nc) == 1:
            if isinstance(nc[0], KetBase):
                return Mul(*c) * InnerProduct(self, nc[0])
            elif isinstance(nc[0], OuterProduct):
                return Mul(*c) * InnerProduct(self, nc[0].ket) * nc[0].bra

        return super().__mul__(other)

class DiscreteOuterProduct(OuterProduct):
    def __mul__(self, other):
        c, nc = other.args_cnc()
        if len(nc) == 1:
            if isinstance(nc[0], KetBase):
                return Mul(*c) * (self.bra * nc[0]) * self.ket
            elif isinstance(nc[0], OuterProduct):
                return Mul(*c) * (self.bra * nc[0].ket) * (self.ket * nc[0].bra)

        return super().__mul__(other)

    def _eval_commutator_DiscreteOuterProduct(self, other, **hints):
        return ((self.bra * other.ket) * (self.ket * other.bra)
                - (other.bra * self.ket) * (other.ket * self.bra))

    def _eval_power(self, exponent):
        if not (getattr(sympify(exponent), 'is_integer', False) and exponent > 0):
            raise ValueError('Operator power defined only for positive integral exponent')

        if (self.bra * self.ket) is S.One:
            return self
        else:
            return S.Zero

    def _apply_from_right_to(self, lhs, **options):
        if isinstance(lhs, BraBase):
            return (lhs * self.ket) * self.bra
        elif isinstance(lhs, OuterProduct):
            return (lhs.bra * self.ket) * (lhs.ket * self.bra)
        raise NotImplementedError(f'Cannot apply OuterProduct to {type(lhs)} from right')



def ketbra(lket, lbra):
    if not isinstance(lket, tuple):
        lket = (lket,)
    if not isinstance(lbra, tuple):
        lbra = (lbra,)
    return DiscreteOuterProduct(DiscreteKet(*lket), DiscreteBra(*lbra))


def get_power(term, param):
    param_power = Order(term, param).args[0]
    if param_power in (S.Zero, S.One):
        return 0
    elif param_power == param:
        return 1
    return param_power.args[1]


def truncate_at_order(expr, param, max_order):
    return _select_by_order(expr, param, lambda term, param: get_power(term, param) <= max_order)


def get_expr_at_order(expr, param, order):
    subexpr = _select_by_order(expr, param, lambda term, param: get_power(term, param) == order)
    return (subexpr / (param ** order)).expand()


def _select_by_order(expr, param, predicate):
    if isinstance(expr, list):
        terms = expr
    else:
        expr = expr.expand()
        if isinstance(expr, Add):
            terms = list(expr.args)
        else:
            terms = [expr]

    result = S.Zero
    for term in terms:
        if predicate(term, param):
            result += term

    return result


def pauli_ops(dim, label=None):
    def _ketbra(i, j):
        if label is None:
            return ketbra(i, j)
        else:
            return ketbra((i, label), (j, label))

    matrices = []

    mat = sum((_ketbra(k, k) for k in range(dim)), S.Zero)
    mat *= sqrt(Rational(2, dim))
    matrices.append(mat)

    for ishell in range(1, dim):
        for ipos in range(ishell):
            matrices.append(_ketbra(ipos, ishell) + _ketbra(ishell, ipos))
            matrices.append(I * (-_ketbra(ipos, ishell) + _ketbra(ishell, ipos)))

        mat = sum((_ketbra(k, k) for k in range(ishell)), S.Zero)
        mat += -ishell * _ketbra(ishell, ishell)
        mat *= sqrt(Rational(2, ishell * (ishell + 1)))
        matrices.append(mat)

    return matrices
