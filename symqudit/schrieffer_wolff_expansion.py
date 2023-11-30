from sympy import Add, I, Mul, Pow, Rational, S, diff, exp, factorial, symbols
from sympy.physics.quantum import Commutator, HermitianOperator

from .common import get_power, truncate_at_order


time = symbols('t', real=True)

class SWExpansion:
    def __init__(self):
        self.g_n = {}
        self.gdot_n = {}

    def expand(self, expr, param, max_order):
        r"""Expand e^{iG}[expr - idt]e^{-iG} in powers of param.

        e^{A} B e^{-A} = sum_{n=0}^{infty} 1/n! Cn[A](B)
        where Cn[A](B) = [A, [A, [A, [..., B]]]] (n-nested)
        e^{A} (-idt) e^{-A} = -sum_{n=0}^{infty} i^n/(n+1)! Cn[A](Adot)
        """
        for n in range(1, max_order + 1):
            if n not in self.g_n:
                self.g_n[n] = HermitianOperator(f'G_{n}')
            if n not in self.gdot_n:
                self.gdot_n[n] = HermitianOperator(fr'\dot{{G}}_{n}')

        leading_order = get_power(expr, param)
        h_term = truncate_at_order(expr, param, max_order)
        for ncom in range(1, max_order - leading_order + 1):
            h_term += (Rational(1, factorial(ncom))
                       * self._nested_sw_commutator(expr, param, max_order, ncom, leading_order))

        gdot = sum(((param ** k) * self.gdot_n[k] for k in range(1, max_order + 1)))
        dt_term = -gdot
        for ncom in range(1, max_order):
            dt_term -= (Rational(1, factorial(ncom + 1))
                        * self._nested_sw_commutator(gdot, param, max_order, ncom, 1))

        return h_term + dt_term

    def _nested_sw_commutator(self, expr, param, max_order, ncom, leading_order):
        if max_order - leading_order - ncom < 0:
            return S.Zero

        # there will be ncom G operators in the nested commutator
        # -> maximum power of λ in G is max_order - leading_order - ncom + 1
        sw_gen = sum(((param ** k) * self.g_n[k]
                      for k in range(1, max_order - leading_order - ncom + 2)),
                     S.Zero)
        # Expand the expr and convert to a list of terms
        expanded = ((I ** ncom) * expr).expand()
        if isinstance(expanded, Add):
            com_terms = list(expanded.args)
        else:
            com_terms = [expanded]

        # Nested commutator - for each term in com_terms, compute the commutator with G and expand
        for _ in range(ncom):
            expanded = sum((Commutator(sw_gen, term).expand(commutator=True) for term in com_terms),
                           S.Zero)
            if isinstance(expanded, Add):
                com_terms = list(expanded.args)
            else:
                com_terms = [expanded]

        # Truncate the final expansion up to max_order
        return truncate_at_order(com_terms, param, max_order)


def integrate_exp_term(term):
    omegas = []
    factors = []
    for factor in term.args:
        if isinstance(factor, exp) and time in factor.free_symbols:
            omegas.append(-I * diff(factor, time).subs({time: 0}))
        else:
            factors.append(factor)

    omega = Add(*omegas)
    coeff = Mul(*factors)
    return -I * (coeff * exp(I * omega * time) - coeff) / omega


def integrate_expr(dot_expr):
    return Add(*[integrate_exp_term(term) for term in dot_expr.args])
