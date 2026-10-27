"""Core algebra for HL-PATP, implemented as the two-sided PATP.

The construction follows ``Spec_HLPATP.md``.  It keeps the frozen PATP
exponent map ``p_i(alpha)`` and splits each ladder rung onto the two half-lines
with independent controls ``alpha_plus`` and ``alpha_minus``.

Design rules enforced here:

* the exponent map is *not* re-parameterised; the one-parameter Form-B ladder is
  recovered exactly on the diagonal ``alpha_plus == alpha_minus`` with equal
  weights, so the family is a genuine extension rather than a new basis class;
* one-sided moments ``nu_q^+`` and ``nu_q^-`` are the atoms.  The symmetric and
  signed moments of the parent apparatus are their sum and difference, so no
  separate moment machinery is introduced;
* the cross-block of the centred Gram matrix is an exact outer product of the
  one-sided means (the half-line supports are disjoint), and this is asserted in
  the tests rather than assumed;
* at the exact point ``alpha_plus == alpha_minus == 1/2`` every rung on a
  given half-line coincides.  The full dictionary therefore has rank two for
  every ``S >= 2`` (under a nondegenerate two-sided law), not ``2S-2``.  The
  canonical implementation keeps one column per half-line and drops the
  redundant linear anchor.  Unlike the one-parameter ladder, the surviving
  system is not generally OLS.

The module contains two deliberately separate moment backends: population
quadrature for oracle calculations and sample one-sided moments for feasible
plug-in calculations.  Callers must preserve that label boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import integrate

try:
    from aepd_core import aepd_moments, aepd_pdf
except ModuleNotFoundError:  # pragma: no cover - direct execution from scripts/
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aepd_core import aepd_moments, aepd_pdf


# --------------------------------------------------------------------------
# Frozen PATP exponent map (decision D2).  Do not re-parameterise.
# --------------------------------------------------------------------------

def exponent(index: int, alpha: float) -> float:
    """Return ``p_i(alpha) = 1/i + (4-i-3/i) alpha + (2i-4+2/i) alpha^2``."""
    if index < 1:
        raise ValueError("ladder index must be >= 1")
    i = float(index)
    return 1.0 / i + (4.0 - i - 3.0 / i) * alpha + (2.0 * i - 4.0 + 2.0 / i) * alpha**2


#: Sample-moment regularity boundary for the derivative moment ``nu_{p-1}``.
#: Root-``N`` plug-in needs ``E|xi|^{2(p-1)} < inf``, i.e. ``p > 1/2`` under a
#: density that is positive and bounded near zero.
REGULARITY_EXPONENT_FLOOR = 0.5

#: Half-width of the protected band around the rank-deficient diagonal point.
DEGENERACY_BAND = 0.02


def regular_alpha_floor(order: int) -> float:
    """Smallest ``alpha`` keeping every rung of the ladder root-``N`` regular.

    Solves ``p_i(alpha) > 1/2`` for the binding rung ``i = order``.
    """
    floor = 0.0
    for index in range(2, order + 1):
        a = 2.0 * index - 4.0 + 2.0 / index
        b = 4.0 - index - 3.0 / index
        c = 1.0 / index - REGULARITY_EXPONENT_FLOOR
        if abs(a) < 1e-15:
            root = -c / b if abs(b) > 1e-15 else 0.0
        else:
            disc = b * b - 4.0 * a * c
            root = (-b + np.sqrt(max(disc, 0.0))) / (2.0 * a)
        floor = max(floor, float(root))
    return floor


# --------------------------------------------------------------------------
# One-sided fractional moments
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseLaw:
    """A standardized (mean zero, unit variance) noise density on ``R``."""

    name: str
    pdf: Callable[[float], float]
    support_low: float = -np.inf
    support_high: float = np.inf
    symmetric: bool = False


class OneSidedMoments:
    """Cached one-sided fractional moments of a standardized noise law.

    ``nu(q, +1) = E[|xi|^q 1{xi > 0}]`` and ``nu(q, -1) = E[|xi|^q 1{xi < 0}]``.
    The parent apparatus is recovered by ``nu_total = nu^+ + nu^-`` and
    ``sigma = nu^+ - nu^-``.
    """

    def __init__(self, law: NoiseLaw, quad_limit: int = 400) -> None:
        self.law = law
        self._quad_limit = quad_limit
        self._cache: dict[tuple[float, int], float] = {}

    # -- integration -------------------------------------------------------
    def _integrate(self, q: float, side: int) -> float:
        if q <= -1.0:
            return float("nan")
        pdf = self.law.pdf
        if side > 0:
            lo = 0.0
            hi = min(self.law.support_high, 200.0)
            f = lambda x: (x**q) * pdf(x)
            knots = [0.0, 1e-3, 1e-2, 0.1, 1.0, 5.0, 20.0, hi]
        else:
            lo = max(self.law.support_low, -200.0)
            hi = 0.0
            f = lambda x: ((-x) ** q) * pdf(x)
            knots = [lo, -20.0, -5.0, -1.0, -0.1, -1e-2, -1e-3, 0.0]
        knots = sorted({float(k) for k in knots if lo <= k <= hi} | {lo, hi})
        total = 0.0
        for left, right in zip(knots[:-1], knots[1:]):
            if right <= left:
                continue
            value, _ = integrate.quad(
                f, left, right, limit=self._quad_limit, epsabs=1e-13, epsrel=1e-11
            )
            total += value
        return float(total)

    def nu(self, q: float, side: int) -> float:
        key = (round(float(q), 12), int(np.sign(side)))
        if key not in self._cache:
            self._cache[key] = self._integrate(float(q), int(np.sign(side)))
        return self._cache[key]

    # -- parent-apparatus views -------------------------------------------
    def nu_total(self, q: float) -> float:
        return self.nu(q, +1) + self.nu(q, -1)

    def sigma(self, q: float) -> float:
        return self.nu(q, +1) - self.nu(q, -1)

    def raw_moment(self, k: int) -> float:
        """Integer raw moment ``E[xi^k]``."""
        sign = 1.0 if k % 2 == 0 else -1.0
        return self.nu(float(k), +1) + sign * self.nu(float(k), -1)

    def c2(self) -> float:
        return self.nu_total(2.0)

    def positive_mass(self) -> float:
        lo = max(self.law.support_low, -200.0)
        hi = min(self.law.support_high, 200.0)
        value, _ = integrate.quad(self.law.pdf, 0.0, hi, limit=self._quad_limit)
        return float(value)


class SampleOneSidedMoments:
    """Residual plug-in counterpart of :class:`OneSidedMoments`.

    Exposes the same ``nu`` / ``nu_total`` / ``sigma`` / ``raw_moment`` / ``c2``
    interface, so every system constructor in this module works unchanged for
    both the oracle and the feasible layer.  This is the only place where sample
    information enters the apparatus.

    Negative-order moments are the delicate part.  ``nu(q, side)`` with
    ``q <= -1`` does not exist even in population, and ``2q <= -1`` means the
    sample estimator has infinite variance under a density positive at zero.
    Both conditions are reported through :meth:`order_is_regular` rather than
    silently smoothed; callers are expected to restrict the exponent grid.
    """

    def __init__(self, residuals: np.ndarray, center: bool = True) -> None:
        values = np.asarray(residuals, dtype=float)
        values = values[np.isfinite(values)]
        if center:
            values = values - values.mean()
        self._values = values
        self._positive = values[values > 0.0]
        self._negative = -values[values < 0.0]
        self._n = max(values.size, 1)
        self._cache: dict[tuple[float, int], float] = {}

    @property
    def sample_size(self) -> int:
        return int(self._values.size)

    @property
    def residuals(self) -> np.ndarray:
        return self._values

    @staticmethod
    def order_is_regular(q: float) -> bool:
        """Root-``N`` regularity of the sample estimator of ``nu_q``."""
        return bool(2.0 * q > -1.0)

    def _side_values(self, side: int) -> np.ndarray:
        return self._positive if side > 0 else self._negative

    def nu(self, q: float, side: int) -> float:
        key = (round(float(q), 12), int(np.sign(side)))
        if key not in self._cache:
            magnitudes = self._side_values(int(np.sign(side)))
            if magnitudes.size == 0:
                value = 0.0
            else:
                if q < 0.0:
                    # a residual exactly at zero would otherwise produce inf
                    magnitudes = magnitudes[magnitudes > 0.0]
                    if magnitudes.size == 0:
                        value = float("nan")
                        self._cache[key] = value
                        return value
                value = float(np.sum(magnitudes**q) / self._n)
            self._cache[key] = value
        return self._cache[key]

    def nu_total(self, q: float) -> float:
        return self.nu(q, +1) + self.nu(q, -1)

    def sigma(self, q: float) -> float:
        return self.nu(q, +1) - self.nu(q, -1)

    def raw_moment(self, k: int) -> float:
        return float(np.mean(self._values**k))

    def c2(self) -> float:
        return float(np.mean(self._values**2))

    def positive_mass(self) -> float:
        return float(self._positive.size / self._n)


# --------------------------------------------------------------------------
# Estimating systems.  Each returns (F, b, labels).
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Component:
    """One centred basis column.

    ``kind`` is one of ``linear``, ``half``, ``signed``, ``power``.  ``side`` is
    ``+1``/``-1`` for half-line columns and ``0`` otherwise.  Keeping the basis
    as data rather than as code lets the same description drive both the
    population matrix and the evaluation on residuals, so the two can never
    drift apart.
    """

    kind: str
    exponent: float
    side: int
    label: str


def component_mean(moments, component: Component) -> float:
    if component.kind == "linear":
        return 0.0
    if component.kind == "half":
        return component.side * moments.nu(component.exponent, component.side)
    if component.kind == "signed":
        return moments.sigma(component.exponent)
    if component.kind == "power":
        return moments.raw_moment(int(component.exponent))
    raise ValueError(f"unknown component kind {component.kind!r}")


def component_sensitivity(moments, component: Component) -> float:
    """``E[d phi / d xi]`` for one column."""
    if component.kind == "linear":
        return 1.0
    p = component.exponent
    if component.kind == "half":
        return p * moments.nu(p - 1.0, component.side)
    if component.kind == "signed":
        return p * moments.nu_total(p - 1.0)
    if component.kind == "power":
        k = int(p)
        return k * moments.raw_moment(k - 1)
    raise ValueError(f"unknown component kind {component.kind!r}")


def _raw_cross_moment(moments, a: Component, b: Component) -> float:
    """``E[phi_a^raw * phi_b^raw]`` before centring."""
    ka, kb = a.kind, b.kind
    if ka == "linear" and kb == "linear":
        return moments.c2()
    if ka == "linear":
        a, b = b, a
        ka, kb = kb, ka
    if kb == "linear":
        if ka == "half":
            return moments.nu(a.exponent + 1.0, a.side)
        if ka == "signed":
            return moments.nu_total(a.exponent + 1.0)
        if ka == "power":
            return moments.raw_moment(int(a.exponent) + 1)
    if ka == "half" and kb == "half":
        if a.side != b.side:
            return 0.0  # disjoint half-line supports
        return moments.nu(a.exponent + b.exponent, a.side)
    if ka == "half" or kb == "half":
        half, other = (a, b) if ka == "half" else (b, a)
        if other.kind == "signed":
            return moments.nu(half.exponent + other.exponent, half.side)
        if other.kind == "power":
            k = int(other.exponent)
            return (half.side ** (k + 1)) * moments.nu(half.exponent + k, half.side)
    if ka == "signed" and kb == "signed":
        return moments.nu_total(a.exponent + b.exponent)
    if ka == "signed" or kb == "signed":
        signed, power = (a, b) if ka == "signed" else (b, a)
        k = int(power.exponent)
        order = signed.exponent + k
        return moments.nu_total(order) if k % 2 == 1 else moments.sigma(order)
    if ka == "power" and kb == "power":
        return moments.raw_moment(int(a.exponent) + int(b.exponent))
    raise ValueError(f"unhandled component pair {ka!r}/{kb!r}")


def evaluate_components(
    residuals: np.ndarray, components: Sequence[Component], means: Sequence[float]
) -> np.ndarray:
    """Return the centred basis matrix with one row per component."""
    xi = np.asarray(residuals, dtype=float)
    rows = []
    for component, mean in zip(components, means):
        if component.kind == "linear":
            row = xi.copy()
        elif component.kind == "half":
            side = component.side
            magnitude = np.maximum(side * xi, 0.0)
            row = side * np.power(magnitude, component.exponent, where=magnitude > 0.0,
                                  out=np.zeros_like(magnitude))
        elif component.kind == "signed":
            magnitude = np.abs(xi)
            row = np.sign(xi) * np.power(magnitude, component.exponent,
                                         where=magnitude > 0.0,
                                         out=np.zeros_like(magnitude))
        elif component.kind == "power":
            row = xi ** int(component.exponent)
        else:
            raise ValueError(f"unknown component kind {component.kind!r}")
        rows.append(row - mean)
    return np.vstack(rows)


@dataclass(frozen=True)
class System:
    F: np.ndarray
    b: np.ndarray
    c2: float
    labels: tuple[str, ...]
    dropped_anchor: bool = False
    components: tuple[Component, ...] = ()
    means: tuple[float, ...] = ()

    @property
    def dimension(self) -> int:
        return len(self.b)


def build_system(moments, components: Sequence[Component], dropped_anchor: bool = False) -> System:
    """Assemble ``F`` and ``b`` for an arbitrary list of components."""
    n = len(components)
    means = [component_mean(moments, component) for component in components]
    F = np.empty((n, n))
    b = np.empty(n)
    for j, component in enumerate(components):
        b[j] = component_sensitivity(moments, component)
        for k in range(j, n):
            raw = _raw_cross_moment(moments, component, components[k])
            F[j, k] = F[k, j] = raw - means[j] * means[k]
    return System(
        F=F,
        b=b,
        c2=moments.c2(),
        labels=tuple(component.label for component in components),
        dropped_anchor=dropped_anchor,
        components=tuple(components),
        means=tuple(means),
    )


def two_sided_system(
    moments: OneSidedMoments,
    alpha_plus: float,
    alpha_minus: float,
    order: int = 2,
    band: float = DEGENERACY_BAND,
) -> System:
    """Build the HL-PATP system.

    Its generic dimension is ``2*order - 1``.  At the *exact* double midpoint
    the canonical span has dimension two for every order: one right-half and
    one left-half linear column.  Points merely close to the midpoint retain
    the full dictionary; numerical instability is handled by the condition
    gate rather than by silently changing the span.

    Basis: ``xi``; then for ``i = 2..order``
    ``(xi^+)^{p_i(a+)} - nu^+``  and  ``-(xi^-)^{p_i(a-)} + nu^-``.
    """
    if order < 2:
        raise ValueError("order must be >= 2")
    exact_tolerance = 1e-12
    degenerate = (
        band > 0.0
        and abs(alpha_plus - 0.5) < exact_tolerance
        and abs(alpha_minus - 0.5) < exact_tolerance
    )
    components = two_sided_components(alpha_plus, alpha_minus, order, degenerate)
    return build_system(moments, components, dropped_anchor=degenerate)


def two_sided_components(
    alpha_plus: float, alpha_minus: float, order: int = 2, degenerate: bool = False
) -> list[Component]:
    if degenerate:
        # p_i(1/2)=1 for every i.  Keeping all rungs would duplicate the same
        # two functions; the linear anchor is their sum for centred residuals.
        return [
            Component("half", 1.0, +1, "plus_midpoint"),
            Component("half", 1.0, -1, "minus_midpoint"),
        ]
    components: list[Component] = []
    components.append(Component("linear", 1.0, 0, "xi"))
    for i in range(2, order + 1):
        components.append(Component("half", exponent(i, alpha_plus), +1, f"plus_{i}"))
    for i in range(2, order + 1):
        components.append(Component("half", exponent(i, alpha_minus), -1, f"minus_{i}"))
    return components


def patp_system(moments, alpha: float, order: int = 2) -> System:
    """One-parameter Form-B PATP ladder: ``xi``; ``sign(xi)|xi|^{p_i(alpha)}``."""
    return full_line_system(moments, [exponent(i, alpha) for i in range(2, order + 1)])


def full_line_components(exponents: Sequence[float]) -> list[Component]:
    return [Component("linear", 1.0, 0, "xi")] + [
        Component("signed", float(q), 0, f"signed_{q:.4f}") for q in exponents
    ]


def full_line_system(moments, exponents: Sequence[float]) -> System:
    """Signed-parity full-line control: ``xi``; ``sign(xi)|xi|^q - sigma_q``."""
    return build_system(moments, full_line_components(exponents))


def classical_pmm_components(degree: int) -> list[Component]:
    return [Component("linear", 1.0, 0, "xi")] + [
        Component("power", float(k), 0, f"power_{k}") for k in range(2, degree + 1)
    ]


def classical_pmm_system(moments, degree: int) -> System:
    """Classical integer-power PMM: ``xi``; ``xi^k - m_k`` for ``k = 2..degree``."""
    return build_system(moments, classical_pmm_components(degree))


# --------------------------------------------------------------------------
# Variance factor and diagnostics
# --------------------------------------------------------------------------

def variance_factor(system: System, condition_limit: float = 1e10) -> float:
    """Return ``g = 1 / (c2 * b' F^{-1} b)``; ``nan`` if the system is unusable.

    No ridge is applied.  A system failing the condition gate is reported as
    ``nan`` rather than silently regularised, because a ridge would change the
    weights, the estimand and the theoretical ``g``.
    """
    F, b = system.F, system.b
    if not np.all(np.isfinite(F)) or not np.all(np.isfinite(b)):
        return float("nan")
    scale = np.sqrt(np.clip(np.diag(F), 1e-300, None))
    Fs = F / np.outer(scale, scale)
    bs = b / scale
    if np.linalg.cond(Fs) > condition_limit:
        return float("nan")
    try:
        kappa = float(bs @ np.linalg.solve(Fs, bs))
    except np.linalg.LinAlgError:
        return float("nan")
    if not np.isfinite(kappa) or kappa <= 0.0:
        return float("nan")
    return 1.0 / (system.c2 * kappa)


def condition_number(system: System) -> float:
    scale = np.sqrt(np.clip(np.diag(system.F), 1e-300, None))
    return float(np.linalg.cond(system.F / np.outer(scale, scale)))


def minimum_eigenvalue(system: System) -> float:
    scale = np.sqrt(np.clip(np.diag(system.F), 1e-300, None))
    return float(np.linalg.eigvalsh(system.F / np.outer(scale, scale)).min())


def two_sided_regularity(alpha_plus: float, alpha_minus: float, order: int = 2) -> dict:
    """Per-side root-``N`` plug-in regularity of the derivative moments."""
    p_plus = [exponent(i, alpha_plus) for i in range(2, order + 1)]
    p_minus = [exponent(i, alpha_minus) for i in range(2, order + 1)]
    margin = min(min(p_plus), min(p_minus)) - REGULARITY_EXPONENT_FLOOR
    return {
        "min_exponent_plus": float(min(p_plus)),
        "min_exponent_minus": float(min(p_minus)),
        "regularity_margin": float(margin),
        "root_n_regular": bool(margin > 0.0),
    }


# --------------------------------------------------------------------------
# Standardized noise laws (the project's AEPD family plus reference laws)
# --------------------------------------------------------------------------

def standardized_aepd(tau: float, p_left: float, p_right: float, name: str = "") -> NoiseLaw:
    """AEPD standardized to mean zero and unit variance.

    ``tau`` is the AEPD split probability (``alpha`` in ``aepd_core``); it is not
    the HL-PATP control.
    """
    stats_ = aepd_moments(0.0, 1.0, tau, p_left, p_right)
    mean = float(stats_.mean)
    sd = float(np.sqrt(stats_.variance))

    def pdf(x):
        return sd * aepd_pdf(np.asarray(x, dtype=float) * sd + mean, 0.0, 1.0, tau, p_left, p_right)

    symmetric = abs(tau - 0.5) < 1e-12 and abs(p_left - p_right) < 1e-12
    label = name or f"aepd_tau{tau:g}_pl{p_left:g}_pr{p_right:g}"
    return NoiseLaw(name=label, pdf=pdf, symmetric=symmetric)


def reference_laws() -> dict[str, NoiseLaw]:
    """Standardized reference laws used as sanity anchors in the oracle gate."""
    from scipy import stats as sps

    laws: dict[str, NoiseLaw] = {}
    laws["normal"] = NoiseLaw("normal", lambda x: sps.norm.pdf(x), symmetric=True)
    laws["laplace"] = NoiseLaw(
        "laplace", lambda x: sps.laplace.pdf(x, scale=1.0 / np.sqrt(2.0)), symmetric=True
    )
    sd_t = np.sqrt(6.0 / 4.0)
    laws["student_t6"] = NoiseLaw(
        "student_t6", lambda x: sd_t * sps.t.pdf(np.asarray(x) * sd_t, 6), symmetric=True
    )
    return laws
