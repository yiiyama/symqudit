from collections import defaultdict
import numpy as np
from sympy import I, S, symbols, poly, factorial, factorial2
from sympy.core.numbers import Rational
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qapply import qapply
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos
from sympy.physics.quantum.sho1d import LoweringOp, RaisingOp, SHOBra, SHOKet


def RaisingOp__print_contents_latex(self, printer, *args):
    arg = printer._print(self.args[0])
    return '{%s^{\\dagger}}' % arg
RaisingOp._print_contents_latex = RaisingOp__print_contents_latex


class Transmon:
    def __init__(self, name, harmonic_freq, anharm_measure):
        self._name = name
        self._harmonic_freq = harmonic_freq
        self._anharm_measure = anharm_measure

        self._omegah = symbols(f'omega_{self._name}', real=True)
        self._epsilon = symbols(f'epsilon_{self._name}', real=True)
        self._b = LoweringOp(f'b_{self._name}')
        self._bdag = RaisingOp(f'b_{self._name}')

        self._hamiltonian = {0: self._bdag * self._b}
        self._coeffs = defaultdict(dict)
        self._eigenvalue_terms = defaultdict(dict)
        self._eigenstates = defaultdict(dict)

    def evaluate(self, expr):
        return expr.subs({self._omegah: self._harmonic_freq, self._epsilon: self._anharm_measure})

    @property
    def phase_op(self):
        return self._b + self._bdag

    @property
    def charge_op(self):
        return -I * (self._b - self._bdag)

    @property
    def exact_hamiltonian(self):
        return self._omegah * Rational(1, 4) * (self.charge_op ** 2 - 2 / self._epsilon
                                                * cos(sqrt(self._epsilon) * self.phase_op))

    def hamiltonian(self, pert_order, normal_order=True):
        return self._omegah * sum(((self._epsilon ** order)
                                   * self.hamiltonian_term(order, normal_order=normal_order)
                                   for order in range(pert_order + 1)),
                                  S.Zero)

    def eigenvalue(self, level, pert_order):
        return self._omegah * sum(((self._epsilon ** order) * self.eigenvalue_term(level, order)
                                   for order in range(pert_order + 1)),
                                  S.Zero)

    def eigenstate(self, level, pert_order):
        if (state := self._eigenstates[level].get(pert_order)) is not None:
            return state

        # |norm> = |unnorm> * (<unnorm|unnorm>)^(-1/2)
        # with |unnorm> = |n> + |phi_n>, <n|phi_n> = 0,
        # <unnorm|unnorm> = 1 + <phi_n|phi_n>
        # let <phi_n|phi_n> = sum_{p=1,q=1}^{infty} ε^(p+q) <psi^p|psi^q> = ε^2 X
        # X = sum_{l=0}^{infty} ε^{l} sum_{r=1}^{l+1} <psi^r|psi^{l+2-r}>
        # (<unnorm|unnorm>)^(-1/2) = (1 + ε^2 X)^(-1/2) = 1 + (-1/2) ε^2 X + 1/2(-1/2)(-3/2) ε^4 X^2 + ...
        # At each order p in |unnorm>, maximum Taylor expansion order kmax is (pert_order - p) // 2.
        # Within each (p, k), maximum power lmax of ε in X is pert_order - p - 2k.
        unnorm_states = [self.eigenstate_term(level, order) for order in range(pert_order + 1)]

        state = S.Zero
        for order in range(pert_order + 1):
            norm = S.One
            for iexp in range(1, (pert_order - order) // 2 + 1):
                norm_term = S.Zero
                for power in range(pert_order - order - 2 * iexp + 1):
                    s = sum((qapply(Dagger(unnorm_states[ir]) * unnorm_states[power + 2 - ir])
                             for ir in range(1, power + 2)),
                            S.Zero)
                    norm_term += (self._epsilon ** power) * s
                c = factorial2(2 * iexp - 1) / factorial(iexp) / ((-2) ** iexp)
                norm += c * (self._epsilon ** (2 * iexp)) * (norm_term ** iexp)

            for im, coeff in self._eigenstate_coeffs(level, order).items():
                state += (self._truncate_perturbation(coeff * norm, pert_order, leading=order)
                          * SHOKet(im))

        self._eigenstates[level][pert_order] = state
        return state

    def unnormalized_eigenstate(self, level, pert_order):
        return sum(((self._epsilon ** order) * self.eigenstate_term(level, order)
                    for order in range(pert_order + 1)),
                   S.Zero)

    def hamiltonian_term(self, order, normal_order=True):
        """Hamiltonian perturbative expansion term of the given order excluding the omegah factor."""
        if not normal_order:
            if order == 0:
                return self._hamiltonian[0]

            return (Rational((-1) ** order, 2 * factorial(2 * order + 2))
                    * (self.phase_op ** (2 * order + 2)))

        if (hamiltonian := self._hamiltonian.get(order)) is not None:
            return hamiltonian

        hamiltonian = S.Zero
        for im in range(order + 1):
            subh = S.Zero
            for il in range(-im - 1, im + 2):
                term = (self._bdag ** (im + 1 + il)) * (self._b ** (im + 1 - il))
                term *= Rational(1, factorial(im + 1 + il) * factorial(im + 1 - il))
                subh += term
            subh *= Rational(1, 2 ** (order - im + 1) * factorial(order - im))
            hamiltonian += subh
        if order % 2 == 1:
            hamiltonian *= -1

        self._hamiltonian[order] = hamiltonian
        return hamiltonian

    def eigenvalue_term(self, level, order):
        """Energy perturbative expansion term of the given level and order excluding the omegah factor."""
        if (value := self._eigenvalue_terms[level].get(order)) is not None:
            return value

        if order == 0:
            value = self._eigenvalue_terms[level][order] = level
            return value

        value = S.Zero
        for ir in range(order):
            ket = qapply(self.hamiltonian_term(order - ir) * self.eigenstate_term(level, ir))
            value += qapply(SHOBra(level) * ket)
        self._eigenvalue_terms[level][order] = value
        return value

    def _eigenstate_coeffs(self, level, order):
        if (coeffs := self._coeffs[level].get(order)) is not None:
            return coeffs

        self._coeffs[level][order] = coeffs = {}

        if order == 0:
            coeffs[level] = S.One
            return coeffs

        min_level = level
        max_level = level
        for ir in range(order):
            eigenstate_r = self.eigenstate_term(level, ir)
            h_order = order - ir
            hamiltonian = self.hamiltonian_term(h_order)
            energy = self.eigenvalue_term(level, h_order)
            # H(p) contains bdagger^(2p+2) and b^(2p+2)
            # -> update min and max levels
            min_level = max(0, min_level - 2 * h_order - 2)
            max_level += 2 * h_order + 2
            for im in range(min_level, max_level + 1):
                if im == level:
                    continue
                bra_m = SHOBra(im)
                coeff = qapply(bra_m * qapply(hamiltonian * eigenstate_r))
                if ir > 0:
                    coeff -= energy * qapply(bra_m * eigenstate_r)
                coeff /= level - im
                if im in coeffs:
                    coeffs[im] += coeff
                else:
                    coeffs[im] = coeff

        return coeffs

    def eigenstate_term(self, level, order):
        coeffs = self._eigenstate_coeffs(level, order)
        return sum((coeff * SHOKet(level) for level, coeff in coeffs.items()),
                   S.Zero)

    def phase_matrix_element(self, row, col, pert_order):
        return self._matrix_element(self.phase_op, row, col, pert_order)

    def charge_matrix_element(self, row, col, pert_order):
        return self._matrix_element(self.charge_op, row, col, pert_order)

    def _matrix_element(self, op, row, col, pert_order):
        row_state = self.eigenstate(row, pert_order)
        col_state = self.eigenstate(col, pert_order)
        me = qapply(Dagger(row_state) * qapply(op * col_state))
        return self._truncate_perturbation(me, pert_order)

    def _truncate_perturbation(self, expr, max_order, leading=0):
        polynomial = poly(expr, self._epsilon)
        # truncate to pert_order
        trunc_expr = S.Zero
        for (power,), c in polynomial.terms():
            if power + leading > max_order:
                continue
            trunc_expr += (self._epsilon ** (power + leading)) * c

        return trunc_expr
