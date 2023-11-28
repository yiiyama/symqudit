from collections import defaultdict
from sympy import Add, I, Mul, Rational, S, Sum, conjugate, diff, exp, simplify, sin, symbols
from sympy.physics.quantum import Dagger, IdentityOperator, TensorProduct, qapply

from .transmon import Transmon
from .common import ketbra, pauli_ops

Id = IdentityOperator()
time = symbols('t', real=True)


class TwoTransmonHamiltonian:
    def __init__(
        self,
        harmonic_freqs,
        anharm_measures
    ):
        self.qc = Transmon('c', harmonic_freqs[0], anharm_measures[0])
        self.qt = Transmon('t', harmonic_freqs[1], anharm_measures[1])

        self.coupling = symbols('J', real=True, positive=True)
        self.drive_amp = symbols('Omega', real=True, positive=True)
        self.drive_freq = symbols('omega_d', real=True, positive=True)
        self.drive_phase = symbols('phi_d', real=True)

    @property
    def h_0_op(self):
        return (TensorProduct(self.qc.hamiltonian_op, Id)
                + TensorProduct(Id, self.qt.hamiltonian_op))

    def h_0_diag(self, cutoff=S.Infinity):
        return (TensorProduct(self.qc.hamiltonian_diag(cutoff), Id)
                + TensorProduct(Id, self.qt.hamiltonian_diag(cutoff)))

    def u_0_diag(self, cutoff=S.Infinity):
        return TensorProduct(self.qc.evolution_diag(cutoff), self.qt.evolution_diag(cutoff))

    @property
    def h_int_op(self):
        drive_fn = self.drive_amp * sin(self.drive_freq * time+ self.drive_phase)
        return (self.coupling * TensorProduct(self.qc.charge_op, self.qt.charge_op)
                + drive_fn * TensorProduct(self.qc.charge_op, Id))

    def h_int_proj(self, cutoff=S.Infinity):
        c_charge = self.qc.charge_proj(cutoff)
        t_charge = self.qt.charge_proj(cutoff)
        drive_fn = self.drive_amp * sin(self.drive_freq * time + self.drive_phase)
        return (self.coupling * TensorProduct(c_charge, t_charge)
                + drive_fn * TensorProduct(c_charge, Id))

    def h_0_ladder(self, epsilon_order, normal_order=True):
        h_0c = self.qc.hamiltonian(epsilon_order, normal_order=normal_order)
        h_0t = self.qt.hamiltonian(epsilon_order, normal_order=normal_order)
        return TensorProduct(h_0c, Id) + TensorProduct(Id, h_0t)

    def h_dirac(self, cutoff=S.Infinity, rwa=True, unit_trans_only=True):
        u0 = self.u_0_diag(cutoff=cutoff)

        if cutoff is S.Infinity:
            return Dagger(u0) * self.h_int_proj(cutoff=cutoff) * u0

        # {(state_c, state_t): coeff}
        u0_bystate = defaultdict(lambda: S.Zero)
        for uterm in u0.expand(tensorproduct=True).args:
            c, nc = uterm.args_cnc()
            c_op, t_op = nc[0].args
            u0_bystate[(c_op.ket.args, t_op.ket.args)] += Mul(*c)

        # {state_c: coeff}
        u0_c_bystate = defaultdict(lambda: S.Zero)
        for uterm in self.qc.evolution_diag(cutoff).expand().args:
            c, nc = uterm.args_cnc()
            c_op = nc[0]
            u0_c_bystate[c_op.ket.args] += Mul(*c)

        c_charge = self.qc.charge_proj(cutoff)
        t_charge = self.qt.charge_proj(cutoff)

        # {(ket_c, ket_t): {(bra_c, bra_t): coeff}}
        hcpl_byket = defaultdict(lambda: defaultdict(lambda: S.Zero))
        for hterm in TensorProduct(c_charge, t_charge).expand(tensorproduct=True).args:
            c, nc = hterm.args_cnc()
            # nc = [TensorProduct(c_outer, t_outer)]
            c_op, t_op = nc[0].args
            diffprod = (c_op.ket.args[0] - c_op.bra.args[0]) * (t_op.ket.args[0] - t_op.bra.args[0])
            if rwa and diffprod > 0:
                # State increment / decrement in the same direction
                continue
            if unit_trans_only and abs(diffprod) != 1:
                # c or t transition is not 1
                continue
            ket_key = (c_op.ket.args, t_op.ket.args)
            bra_key = (c_op.bra.args, t_op.bra.args)
            hcpl_byket[ket_key][bra_key] += Mul(self.coupling, *c)

        # {ket_c: {bra_c: coeff}}
        hdrv_byket = defaultdict(lambda: defaultdict(lambda: S.Zero))
        if rwa:
            drive_fn_pos = (-I * self.drive_amp * Rational(1, 2)
                            * exp(I * (self.drive_freq * time + self.drive_phase)))
            drive_fn_neg = conjugate(drive_fn_pos)
        else:
            drive_fn = self.drive_amp * sin(self.drive_freq * time + self.drive_phase)

        for hterm in c_charge.expand().args:
            c, nc = hterm.args_cnc()
            c_op = nc[0]
            if unit_trans_only and abs(c_op.ket.args[0] - c_op.bra.args[0]) != 1:
                continue
            if rwa:
                if c_op.ket.args[0] < c_op.bra.args[0]:
                    drive_fn = drive_fn_pos
                else:
                    drive_fn = drive_fn_neg

            hdrv_byket[c_op.ket.args][c_op.bra.args] += Mul(drive_fn, *c)

        res_terms = []
        for lhs_state, ufact in u0_bystate.items():
            udagfact = conjugate(ufact)
            for rhs_state, hfact in hcpl_byket[lhs_state].items():
                if (ufact := u0_bystate.get(rhs_state)) is None:
                    continue
                op = TensorProduct(ketbra(lhs_state[0], rhs_state[0]),
                                   ketbra(lhs_state[1], rhs_state[1]))
                if lhs_state == rhs_state:
                    res_terms.append(hfact * op)
                else:
                    res_terms.append(((udagfact * ufact).doit() * hfact) * op)

        for lhs_c_state, ufact in u0_c_bystate.items():
            udagfact = conjugate(ufact)
            for rhs_c_state, hfact in hdrv_byket[lhs_c_state].items():
                if (ufact := u0_c_bystate.get(rhs_c_state)) is None:
                    continue
                op = TensorProduct(ketbra(lhs_c_state, rhs_c_state), Id)
                if lhs_c_state == rhs_c_state:
                    res_terms.append(hfact * op)
                else:
                    res_terms.append(((udagfact * ufact).doit() * hfact) * op)

        return Add(*res_terms)

    def subs_delta(self, expr):
        """Replace omega differences in h_dirac(rwa=True) with Delta expressions."""
        time = symbols('t', real=True)
        result_terms = []
        for term in expr.expand(tensorproduct=True).doit().args:
            factors = []
            ind1, ind2, sgn = None, None, None
            for factor in term.args:
                if isinstance(factor, exp) and time in (free_symbols := factor.free_symbols):
                    exponent = factor.args[0]
                    if self.qc._omegah in free_symbols:
                        ind1 = next(elem for elem in exponent.args
                                    if isinstance(elem, self.qc._energygap)).args[0]
                        if -1 in exponent.args:
                            sgn = -1
                        else:
                            sgn = 1
                    elif self.qt._omegah in free_symbols:
                        ind2 = next(elem for elem in exponent.args
                                    if isinstance(elem, self.qt._energygap)).args[0]
                    elif self.drive_freq in free_symbols:
                        ind2 = 'd'
                    else:
                        raise ValueError(f'Unhandled exponential {factor}')
                    continue

                factors.append(factor)

            delta = symbols(fr'{{\Delta^{ind1}_{ind2}}}', real=True)
            result_terms.append(Mul(exp(sgn * I * delta * time), *factors))

        return Add(*result_terms)

    def paulis(self, cdim, tdim):
        ops = {}
        for ic, cop in enumerate(pauli_ops(cdim, self.qc.name)):
            for it, top in enumerate(pauli_ops(tdim, self.qt.name)):
                ops[(ic, it)] = TensorProduct(cop, top).expand(tensorproduct=True)

        return ops

    def pauli_components(self, expr, cdim, tdim):
        """Coefficients of 1/2 lambda_i x lambda_j"""
        expr_dict = to_dict(expr)

        components = {}
        for idx, op in self.paulis(cdim, tdim).items():
            prod = dict_product(to_dict(op), expr_dict)
            component = []
            for ket, bra_dict in prod.items():
                if (coeff := bra_dict.get(ket)) is not None:
                    component.append(coeff)

            if component:
                components[idx] = (Add(*component) / 2).expand()

        return components


def sort_block_diagonal(expr):
    b_terms = []
    n_terms = []
    time = symbols('t', real=True)

    for term in expr.expand(tensorproduct=True).doit().args:
        c, nc = term.args_cnc()
        c_op, t_op = nc[0].args
        if c_op.ket.args[0] == c_op.bra.args[0]:
            exponents = [factor.args[0] for factor in c
                         if isinstance(factor, exp) and time in factor.free_symbols]
            if Add(*exponents).expand().doit() is S.Zero:
                b_terms.append(term)
                continue
        else:
            n_terms.append(term)

    return Add(*b_terms), Add(*n_terms)


def organize(expr):
    terms = defaultdict(lambda: defaultdict(lambda: S.Zero))
    for term in expr.expand(power_exp=False, tensorproduct=True, commutator=True).doit().args:
        c, nc = term.args_cnc()
        tprod = nc[0]
        terms[tprod.args[0]][tprod.args[1]] += Mul(*c)

    full_result = []
    for cop, t_terms in terms.items():
        c_result = []
        for top, factor in t_terms.items():
            #c_result.append(simplify(factor) * top)
            c_result.append(factor * top)

        full_result.append(TensorProduct(cop, Add(*c_result)))

    return Add(*full_result)


def to_dict(add_expr):
    """Convert an Add expression with terms of form C*(OuterProduct)x(OuterProduct or Id) to a dict."""
    terms = defaultdict(lambda: defaultdict(lambda: S.Zero))
    for term in add_expr.args:
        c, nc = term.args_cnc()
        c_op, t_op = nc[0].args
        c_ket, c_bra = tuple(arg.args for arg in c_op.args)
        if isinstance(t_op, IdentityOperator):
            ket = (c_ket, None)
            bra = (c_bra, None)
        else:
            ket = (c_ket, t_op.args[0].args)
            bra = (c_bra, t_op.args[1].args)

        terms[ket][bra] += Mul(*c)

    return terms


def from_dict(terms):
    result = []
    for ket, bra_dict in terms.items():
        for bra, coeff in bra_dict.items():
            if ket[1] is None:
                result.append(coeff * TensorProduct(ketbra(ket[0], bra[0]), Id))
            else:
                result.append(coeff * TensorProduct(ketbra(ket[0], bra[0]), ketbra(ket[1], bra[1])))

    return Add(*result)


def dict_product(lhs, rhs):
    """Operator product of dicts returned by to_dict."""
    rhs_ct = defaultdict(dict)
    for (c_ket, t_ket), bra_dict in rhs.items():
        rhs_ct[c_ket][t_ket] = bra_dict

    out = defaultdict(lambda: defaultdict(lambda: S.Zero))
    for lhs_ket, lhs_bra_dict in lhs.items():
        for (c_bra, t_bra), lhs_coeff in lhs_bra_dict.items():
            if (rhs_t := rhs_ct.get(c_bra)) is None:
                continue
            if lhs_ket[1] is None:
                for t_ket, rhs_bra_dict in rhs_t.items():
                    for rhs_bra, rhs_coeff in rhs_bra_dict.items():
                        out[(lhs_ket[0], t_ket)][(rhs_bra)] += lhs_coeff * rhs_coeff
            else:
                if (rhs_bra_dict := rhs_t.get(t_bra)) is not None:
                    for rhs_bra, rhs_coeff in rhs_bra_dict.items():
                        out[lhs_ket][rhs_bra] += lhs_coeff * rhs_coeff
                if (rhs_bra_dict := rhs_t.get(None)) is not None:
                    for rhs_bra, rhs_coeff in rhs_bra_dict.items():
                        out[lhs_ket][(rhs_bra[0], t_bra)] += lhs_coeff * rhs_coeff

    return out


def pauli_components(hamiltonian_dict, cdim, tdim):
    matrices = paulis((cdim, tdim))
    for ic in range(cdim ** 2):
        for it in range(tdim ** 2):
            c_
