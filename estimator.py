"""Pure-Python FOLEAGE PCG complexity estimator.

Public entry points:
  - expect_cost(c, t, s, q=4, ...)  -> float (bits of security)
  - find_t(c, s, security_parameter=128, q=4, ...) -> int (required t)

ISD complexities are computed in log2-space throughout to avoid overflow
on huge intermediate quantities; the weight-distribution polynomials use
exact Fractions to keep the final fold-weight probability accurate to
floating-point precision.
"""
from __future__ import annotations

import math
from fractions import Fraction
from math import comb, inf, lgamma, log2
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Aggressivity tuning. Mutate via set_aggressivity.
# ---------------------------------------------------------------------------
conservative = True
csplit = False


def set_aggressivity(
    *, conservative_: bool | None = None, csplit_: bool | None = None
) -> None:
    """Override the module-level aggressivity flags."""
    global conservative, csplit
    if conservative_ is not None:
        conservative = conservative_
    if csplit_ is not None:
        csplit = csplit_


# ---------------------------------------------------------------------------
# Numeric helpers.
# ---------------------------------------------------------------------------
_LN2 = math.log(2)


def _log2_pos(x: int | float | Fraction) -> float:
    """log2 of a positive int / Fraction / float without overflowing to float first."""
    if isinstance(x, Fraction):
        if x.numerator <= 0:
            raise ValueError(f"log2 of non-positive Fraction: {x}")
        return log2(x.numerator) - log2(x.denominator)
    if isinstance(x, int):
        if x <= 0:
            raise ValueError(f"log2 of non-positive int: {x}")
        return log2(x)
    if x <= 0:
        raise ValueError(f"log2 of non-positive float: {x}")
    return log2(x)


def _is_prime_power(n: int) -> bool:
    if n < 2:
        return False
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % p == 0:
            while n % p == 0:
                n //= p
            return n == 1
    return True


def _gaussian_binomial(n: int, k: int, q: int) -> int:
    """[n choose k]_q = prod_{i=0}^{k-1} (q^(n-i) - 1) / (q^(i+1) - 1)."""
    if k < 0 or k > n:
        return 0
    num, den = 1, 1
    for i in range(k):
        num *= q ** (n - i) - 1
        den *= q ** (i + 1) - 1
    return num // den


def _newton(
    f: Callable[[float], float],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 200,
    h: float = 1e-7,
) -> float:
    """Newton's method with finite-difference derivative (matches scipy.optimize.newton)."""
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = (f(x + h) - f(x - h)) / (2 * h)
        if dfx == 0.0:
            raise RuntimeError("Newton's method: zero derivative")
        x -= fx / dfx
    return x


def _logsumexp2(xs: list[float]) -> float:
    """log2(sum(2^x for x in xs)), tolerant of -inf entries."""
    finite = [x for x in xs if x != -inf]
    if not finite:
        return -inf
    m = max(finite)
    return m + log2(sum(2.0 ** (x - m) for x in finite))


def _log2_binom(n: int, k: int) -> float:
    """log2 of an integer binomial coefficient (exact via comb, then log2)."""
    if n < 0 or k < 0 or k > n:
        return -inf
    if k == 0 or k == n:
        return 0.0
    return _log2_pos(comb(n, k))


def _log2_gen_binom(a: float, b: float) -> float:
    """Gamma-extended log2 binomial."""
    return (lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)) / _LN2


# ---------------------------------------------------------------------------
# Polynomials (truncated). Two flavours: Fraction coefficients for exact
# weight-enumerator arithmetic, float coefficients for the final convolution.
# ---------------------------------------------------------------------------
def _polyf_mul(a: list[Fraction], b: list[Fraction], max_deg: int) -> list[Fraction]:
    out_len = min(len(a) + len(b) - 1, max_deg + 1) if a and b else 0
    if out_len <= 0:
        return []
    out = [Fraction(0)] * out_len
    for i, ai in enumerate(a):
        if i > max_deg or ai == 0:
            continue
        b_limit = min(len(b), out_len - i)
        for j in range(b_limit):
            out[i + j] += ai * b[j]
    return out


def _polyf_pow(p: list[Fraction], e: int, max_deg: int) -> list[Fraction]:
    result: list[Fraction] = [Fraction(1)]
    base = list(p)
    while e:
        if e & 1:
            result = _polyf_mul(result, base, max_deg)
        e >>= 1
        if e:
            base = _polyf_mul(base, base, max_deg)
    return result


def _poly_at_frac(p: list[Fraction], idx: int) -> Fraction:
    return p[idx] if 0 <= idx < len(p) else Fraction(0)


def _poly_mul(a: list[float], b: list[float], max_deg: int) -> list[float]:
    out_len = min(len(a) + len(b) - 1, max_deg + 1) if a and b else 0
    if out_len <= 0:
        return []
    out = [0.0] * out_len
    for i, ai in enumerate(a):
        if i > max_deg or ai == 0.0:
            continue
        b_limit = min(len(b), out_len - i)
        for j in range(b_limit):
            out[i + j] += ai * b[j]
    return out


def _poly_pow(p: list[float], e: int, max_deg: int) -> list[float]:
    result: list[float] = [1.0]
    base = list(p)
    while e:
        if e & 1:
            result = _poly_mul(result, base, max_deg)
        e >>= 1
        if e:
            base = _poly_mul(base, base, max_deg)
    return result


# ===========================================================================
# ISD algorithms
# ===========================================================================
def _Tgauss(n: int, k: int) -> int:
    """Lower bound on the cost of Gaussian elimination, favoring the attacker."""
    return n * (n - k)


def c_split_loss(n: int, t: int, c: int) -> float:
    """Penalty (in log2) for the error being regularly split into c blocks."""
    if not csplit:
        return 0.0
    return c * _log2_gen_binom(n / c, t / c) - _log2_pos(comb(n, t))


def Prange(t: int, k: int, n: int, full: bool = False, **_: Any):
    """Bit complexity of the classic Prange algorithm."""
    res = _log2_binom(n, t) - _log2_binom(n - k, t) + log2(_Tgauss(n, k))
    if full:
        return res, {}
    return res


def Lee_brickell(
    t: int, k: int, n: int, q: int, p: int, verbose: bool = False, **_: Any
) -> float:
    """Bit complexity of classic Lee-Brickell with p errors in the information set."""
    if verbose:
        print(f"n={n}, k={k}, t={t}, p={p}")
    log2_P_succ = _log2_binom(k, p) + _log2_binom(n - k, t - p) - _log2_binom(n, t)
    if log2_P_succ == -inf:
        return inf
    log2_L = _log2_binom(k, p) + p * (log2(q - 1) if q > 1 else -inf)
    log2_T_gauss = log2(_Tgauss(n, k))
    log2_C_iter = _logsumexp2([log2_T_gauss, log2_L])
    return log2_C_iter - log2_P_succ


def Lee_brickell_opt(
    t: int,
    k: int,
    n: int,
    q: int = 4,
    p_min: int = 0,
    full: bool = False,
    verbose: bool = False,
    **_: Any,
):
    """Find the optimal p for Lee-Brickell; in practice p=2 wins."""
    T_min = Lee_brickell(t, k, n, q, p_min)
    for p in range(p_min + 1, t):
        T_cur = Lee_brickell(t, k, n, q, p)
        if T_cur < T_min:
            T_min, p_min = T_cur, p
    if verbose:
        print(
            f"Lee-Brickell for t={t}, n={n}, k={k} is optimal for p={p_min} --> {T_min}"
        )
    if full:
        return T_min, {"p_min": p_min}
    return T_min


def Stern(
    t: int, k: int, n: int, q: int, p: int, ell: int, verbose: bool = False
) -> float:
    """Bit complexity of q-ary Stern-Dumer."""
    if t == 0:
        return 0.0

    half_kl = (k + ell) // 2
    half_p = p // 2
    log2_q = log2(q)
    log2_qm1 = log2(q - 1) if q > 1 else -inf

    log2_binom_kl_p = _log2_binom(half_kl, half_p)
    if log2_binom_kl_p == -inf:
        return inf

    log2_L1 = log2_binom_kl_p + (p / 2) * log2_qm1
    log2_L = 2 * log2_L1 - ell * log2_q

    nkl = n - k - ell
    log2_nkl = log2(nkl) if nkl > 0 else -inf
    log2_kl = log2(k + ell) if k + ell > 0 else -inf

    if conservative:
        # L*(n-k-ell)*(k+ell) + max(L_1, L) + L_1
        log2_C_iter = _logsumexp2(
            [log2_L + log2_nkl + log2_kl, max(log2_L1, log2_L), log2_L1]
        )
    else:
        # L*(n-k-ell)*(k+ell) + 2*ell*(k+ell)*L_1 + L*ell
        log2_term2 = (
            log2(2 * ell * (k + ell)) + log2_L1 if ell > 0 and (k + ell) > 0 else -inf
        )
        log2_term3 = log2(ell) + log2_L if ell > 0 else -inf
        log2_C_iter = _logsumexp2(
            [log2_L + log2_nkl + log2_kl, log2_term2, log2_term3]
        )

    log2_T_gauss = log2(_Tgauss(n, k + ell))
    log2_total = _logsumexp2([log2_T_gauss, log2_C_iter])

    log2_P_succ_num = _log2_binom(nkl, t - p) + 2 * log2_binom_kl_p
    if log2_P_succ_num == -inf:
        return inf
    log2_P_succ = log2_P_succ_num - _log2_binom(n, t)

    return log2_total + log2(log2_q) - log2_P_succ


def Stern_opt(
    t: int,
    k: int,
    n: int,
    q: int = 4,
    p_min: int = 0,
    ell_min: int = 0,
    full: bool = False,
    verbose: bool = False,
    **_: Any,
):
    """Sweep (p, ell) for the cheapest Stern run."""
    p_min, ell_min = 0, 0
    T_min = Stern(t, k, n, q, p_min, ell_min)
    if t < 30:
        p_max, ell_max = 8, 30
    elif t < 80:
        p_max, ell_max = 15, 50
    else:
        p_max, ell_max = 15, 100
    for p in range(1, min(t + 1, p_max)):
        for ell in range(1, min(n - k + 1, ell_max)):
            T_cur = Stern(t, k, n, q, p, ell)
            if T_cur < T_min:
                T_min, p_min, ell_min = T_cur, p, ell
    if verbose:
        print(
            f"Complexity Stern for decoding {t} errors, in an [{n}, {k}]_{q}-code"
            f" is optimal for p={p_min} l={ell_min} ---> {T_min}"
        )
    if full:
        return T_min, {"p_min": p_min, "ell_min": ell_min}
    return T_min


def Optimized_Stern(
    t: int, k: int, n: int, q: int, p: int, ell: int, verbose: bool = False
) -> float:
    """q-ary Stern with the [P09] tweaks."""
    if t == 0:
        return 0.0
    half_k = k // 2
    if t - 2 * p < 0:
        return inf

    log2_q = log2(q)
    log2_qm1 = log2(q - 1) if q > 1 else -inf

    log2_binom_l = _log2_binom(half_k, p)
    log2_binom_r = _log2_binom(k - half_k, p)
    if log2_binom_l == -inf or log2_binom_r == -inf:
        return inf

    log2_L1 = log2_binom_l + p * log2_qm1
    log2_L2 = log2_binom_r + p * log2_qm1
    log2_T_gauss = log2(_Tgauss(n, k + ell))

    # T_lists = (k/2 - p + 1 + L_1 + L_2) * ell
    log2_kp1 = log2(half_k - p + 1) if (half_k - p + 1) > 0 else -inf
    log2_T_lists_inner = _logsumexp2([log2_kp1, log2_L1, log2_L2])
    log2_T_lists = log2_T_lists_inner + (log2(ell) if ell > 0 else -inf)

    # N_cols = L1 * L2 / q^ell
    log2_N_cols = log2_L1 + log2_L2 - ell * log2_q

    # T_checks = (q/(q-1)) * (t - 2p + 1) * 2p * (1 + (q-2)/(q-1)) * N_cols
    factor = (q / (q - 1)) * (t - 2 * p + 1) * 2 * p * (1 + (q - 2) / (q - 1))
    log2_T_checks = (log2(factor) + log2_N_cols) if factor > 0 else -inf

    # P_succ = L1 * L2 * C(n-k-ell, t-2p) / (C(n,t) * (q-1)^(2p))
    log2_C_target = _log2_binom(n - k - ell, t - 2 * p)
    if log2_C_target == -inf:
        return inf
    log2_P_succ = (
        log2_L1
        + log2_L2
        + log2_C_target
        - _log2_binom(n, t)
        - 2 * p * log2_qm1
    )
    if log2_P_succ < log2(1e-53):
        return inf

    log2_C_iter = (
        _logsumexp2([log2_T_gauss, log2_T_lists, log2_T_checks]) + log2(log2_q)
    )
    return log2_C_iter - log2_P_succ


def Optimized_Stern_opt(
    t: int,
    k: int,
    n: int,
    q: int = 4,
    p_min: int = 0,
    ell_min: int = 0,
    full: bool = False,
    verbose: bool = False,
    **_: Any,
):
    """Sweep (p, ell) for the cheapest Optimized_Stern run."""
    if t < 30:
        p_max, ell_max = 15, 30
    elif t < 80:
        p_max, ell_max = 15, 50
    else:
        p_max, ell_max = 15, 100
    p_min, ell_min = 0, 0
    T_min = Optimized_Stern(t, k, n, q, p_min, ell_min)
    p_upper = min(t // 2, k // 2, p_max)
    for p in range(1, p_upper):
        ell_upper = min(n - k, n - k + 2 * p - t, ell_max)
        for ell in range(1, ell_upper):
            T_cur = Optimized_Stern(t, k, n, q, p, ell)
            if T_cur < T_min:
                T_min, p_min, ell_min = T_cur, p, ell
    if verbose:
        print(
            f"Complexity Optimized_Stern for decoding {t} errors, in an "
            f"[{n}, {k}]_{q}-code is optimal for p={p_min} l={ell_min} ---> {T_min}"
        )
    if full:
        return T_min, {"p_min": p_min, "ell_min": ell_min}
    return T_min


def MMT(
    t: int, k: int, n: int, q: int, p: int, ell: int, verbose: bool = False, **_: Any
) -> float:
    """Bit complexity of q-ary MMT."""
    if t == 0:
        return 0.0
    log2_q = log2(q)
    log2_qm1 = log2(q - 1) if q > 1 else -inf

    log2_T_gauss = log2(_Tgauss(n, k + ell))

    log2_P_succ1 = (
        _log2_binom(k + ell, p)
        + _log2_binom(n - k - ell, t - p)
        - _log2_binom(n, t)
    )
    log2_P_succ2 = 4 * _log2_binom((k + ell) // 2, p // 4) - 2 * _log2_binom(
        k + ell, p // 2
    )
    log2_P_succ = log2_P_succ1 + log2_P_succ2
    if log2_P_succ == -inf:
        return inf

    R = comb(p, p // 2)
    if R == 0:
        return inf
    log2_R = log2(R)

    log2_kl_half_q4 = _log2_binom((k + ell) // 2, p // 4)
    log2_L00 = log2_kl_half_q4 + (p // 4) * log2_qm1
    log2_L02 = 2 * log2_kl_half_q4 + (p // 2) * log2_qm1 - log2_R
    log2_L = 4 * log2_kl_half_q4 + p * log2_qm1 - ell * log2_q - log2_R

    if verbose:
        print(f"k={k},l={ell},p={p},q={q}")
        print(
            f"L00=2^{log2_L00},L02=2^{log2_L02},L=2^{log2_L}, "
            f"p_succ=2^{log2_P_succ}"
        )

    nkl = n - k - ell
    log2_nkl = log2(nkl) if nkl > 0 else -inf
    log2_kl = log2(k + ell) if k + ell > 0 else -inf

    if conservative:
        log2_C_iter = _logsumexp2(
            [
                log2_T_gauss,
                log2_L + log2_nkl + log2_kl,
                max(log2_L00, log2_L02, log2_L),
                log2_L00,
            ]
        )
    else:
        log2_max = max(log2_L00, log2_L02, log2_L)
        log2_t3 = log2(3 * ell) + log2_max if ell > 0 else -inf
        log2_t4 = (
            log2(4 * (k + ell) * ell) + log2_L00 if ell > 0 and (k + ell) > 0 else -inf
        )
        log2_C_iter = _logsumexp2(
            [log2_T_gauss, log2_L + log2_nkl + log2_kl, log2_t3, log2_t4]
        )

    return log2_C_iter + log2(log2_q) - log2_P_succ


def MMT_opt(
    t: int,
    k: int,
    n: int,
    q: int = 4,
    p_min: int = 0,
    ell_min: int = 1,
    full: bool = False,
    verbose: bool = False,
    **_: Any,
):
    """Sweep (p, ell) for the cheapest MMT run."""
    ell_min = 1
    p_min = 1
    T_min = MMT(t, k, n, q, p_min, ell_min, verbose=verbose)
    if t < 30:
        p_max, ell_max = 8, 30
    elif t < 160:
        p_max, ell_max = 15, 50
    else:
        p_max, ell_max = 15, 100
    for p in range(1, min(t, p_max) + 1):
        for ell in range(1, min(n - k + p - t + 1, ell_max)):
            T_cur = MMT(t, k, n, q, p, ell, verbose=verbose)
            if T_cur < T_min:
                T_min, p_min, ell_min = T_cur, p, ell
    if verbose:
        print(
            f"Complexity MMT for decoding {t} errors, in an [{n}, {k}]_{q}-code"
            f" is optimal for p={p_min} l={ell_min} ---> {T_min}"
        )
    if full:
        return T_min, {"p_min": p_min, "ell_min": ell_min}
    return T_min


ISDs: dict[str, Callable[..., Any]] = {
    "Prange": Prange,
    "Lee-Brickell": Lee_brickell_opt,
    "Stern": Stern_opt,
    "Optimized_Stern": Optimized_Stern_opt,
    # "MMT": MMT_opt,
}


# ===========================================================================
# Folding analysis
# ===========================================================================
def h_q(q: int, x: float) -> float:
    """q-ary entropy function."""
    log_q = math.log(q)
    return (
        -x * math.log(x / (q - 1)) / log_q - (1 - x) * math.log(1 - x) / log_q
    )


def GV(R: float, q: int = 4) -> float:
    """Gilbert-Varshamov bound for rate R: h_q^{-1}(1 - R)."""
    return _newton(lambda x: h_q(q, x) - (1 - R), 0.1)


def doom_loss(s: int, d: int) -> float:
    """Speedup factor sqrt(|G|) in the DOOM analysis (returns log2)."""
    return -(s / 2) * log2(d)


def prange_c_split_doom(
    c: int, t: int, s: int, q: int = 4, d: int | None = None
) -> float:
    """Lower bound on Prange's cost in the c-split DOOM regime."""
    if d is None:
        d = q - 1
    N = d ** s
    w = c * t
    n = c * N
    k = (c - 1) * N
    return Prange(w, k, n) + doom_loss(s, d) + c_split_loss(n, w, c)


def estimator_init(
    q: int = 4,
    c: int = 4,
    s: int = 16,
    d: int | None = None,
    target_security: int = 128,
) -> int:
    """Find the minimal pre-folding weight that already gives target security."""
    if d is None:
        d = q - 1
    t = 1
    if prange_c_split_doom(c, t, s, q, d) > target_security:
        return t
    while prange_c_split_doom(c, t, s, q, d) < target_security:
        t *= 2
    lo, hi = t // 2, t
    while lo <= hi:
        t = (lo + hi) // 2
        if prange_c_split_doom(c, t, s, q, d) > target_security:
            if prange_c_split_doom(c, t - 1, s, q, d) <= target_security:
                return t
            hi = t - 1
        else:
            lo = t + 1
    return lo


def min_complexity_fold(
    t: int, q: int = 4, c: int = 4, d: int | None = None
) -> float:
    """Best ISD complexity for decoding at distance t close to GV at rate 1-1/c."""
    if d is None:
        d = q - 1
    gv = GV(1 - 1 / c, q)
    s_fold = math.ceil(math.log(t / gv) / math.log(d))
    N = d ** s_fold
    Cmin = inf
    for ISD in ISDs.values():
        T = ISD(c * t, (c - 1) * N, c * N, q=q) + doom_loss(s_fold, d)
        if T < Cmin:
            Cmin = T
    return float(Cmin)


def refine_t(
    q: int = 4,
    c: int = 4,
    s: int = 16,
    t: int | None = None,
    d: int | None = None,
    target_security: int = 128,
) -> int:
    """Refine the lower bound on t using ISD complexities on the folded code."""
    if d is None:
        d = q - 1
    if t is None:
        t = estimator_init(q, c, s, d, target_security)
    while min_complexity_fold(t, q, c, d) < target_security:
        t += 1
    return t


# ---------------------------------------------------------------------------
# Weight enumerators of the [l, l-1]_q parity code and its complement.
# ---------------------------------------------------------------------------
def f_0(l: int, t: int, q: int = 4) -> list[Fraction]:
    """f_0 = (1/q) * ((1 + (q-1)X)^l + (q-1)(1 - X)^l), truncated to deg t."""
    inv_q = Fraction(1, q)
    out = [Fraction(0)] * (t + 1)
    for k in range(t + 1):
        coef = comb(l, k) * ((q - 1) ** k + ((-1) ** k) * (q - 1))
        out[k] = inv_q * coef
    return out


def f_1(l: int, t: int, q: int = 4) -> list[Fraction]:
    """f_1 = ((q-1)/q) * ((1 + (q-1)X)^l - (1 - X)^l), truncated to deg t."""
    factor = Fraction(q - 1, q)
    out = [Fraction(0)] * (t + 1)
    for k in range(t + 1):
        coef = comb(l, k) * ((q - 1) ** k - ((-1) ** k))
        out[k] = factor * coef
    return out


def distribution_weight_fold(
    s: int,
    s_H: int,
    t: int,
    q: int = 4,
    d: int | None = None,
    verbose: bool = False,
) -> list[float]:
    """Probability that the folded error has weight u, for u in [0, t]."""
    if d is None:
        d = q - 1

    fold_s = s - s_H
    n = d ** s
    np = d ** fold_s
    l = d ** s_H

    f0 = f_0(l, t, q)
    f1 = f_1(l, t, q)

    log2_denom = _log2_binom(n, t) + t * (log2(q - 1) if q > 1 else -inf)

    L = [0.0] * (t + 1)
    for u in range(t + 1):
        if verbose:
            print(f"\tComputing the probability that the folded error has weight {u}")
        prod = _polyf_mul(_polyf_pow(f1, u, t), _polyf_pow(f0, np - u, t), t)
        Atu = _poly_at_frac(prod, t)
        if Atu <= 0:
            L[u] = 0.0
            continue
        # Atu = num / den; full ratio is binom(np,u) * Atu / (binom(n,t) * (q-1)^t).
        log2_num = _log2_binom(np, u) + _log2_pos(Atu.numerator)
        log2_full_denom = log2_denom + _log2_pos(Atu.denominator)
        L[u] = 2.0 ** (log2_num - log2_full_denom)
    return L


def distribution_weight_folded_full(
    s: int,
    s_H: int,
    t: int,
    c: int,
    q: int = 4,
    d: int | None = None,
    verbose: bool = False,
) -> list[float]:
    """c-fold convolution of the folded weight distribution. Length c*t+1."""
    if d is None:
        d = q - 1
    L = distribution_weight_fold(s, s_H, t, q, d, verbose)
    return _poly_pow(L, c, c * t)


def ISD_vec(
    c: int,
    t: int,
    fold_s: int,
    q: int = 4,
    d: int | None = None,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """ISD complexity at each weight 0..c*t for every enabled algorithm."""
    if d is None:
        d = q - 1
    n = c * d ** fold_s
    k = (c - 1) * d ** fold_s
    out: dict[str, list[float]] = {}
    for name, ISD in ISDs.items():
        if verbose:
            print(
                f"Computing cost of {name} up to {c*t} errors in an "
                f"[{n}, {k}]_{q} code..."
            )
        L: list[float] = []
        params: dict[str, Any] = {"p_min": 0, "ell_min": 0}
        for w in range(c * t + 1):
            res, params = ISD(w, k, n, q=q, verbose=verbose, full=True, **params)
            L.append(res + doom_loss(fold_s, d) + c_split_loss(n, c * t, c))
        out[name] = L
        if verbose:
            print("...Done")
    return out


def expect_cost(
    c: int,
    t: int,
    s: int,
    q: int = 4,
    r: int = 1,
    d: int | None = None,
    ISD_l: dict[str, list[float]] | None = None,
    verbose: bool = False,
    offset: int = 0,
) -> float:
    """Expected bit-cost of the folding attack."""
    if d is None:
        d = q - 1
    if not _is_prime_power(q):
        raise ValueError("Field size q must be a prime power.")
    if (q - 1) % d != 0:
        raise ValueError("d should divide q-1.")
    R = 1 - 1 / c
    gv = GV(R, q)

    fold_s = math.ceil(math.log(t / gv) / math.log(d)) + offset
    s_H = s - fold_s

    Cost_folding = c * d ** s

    if ISD_l is None:
        ISD_l = ISD_vec(c, t, fold_s, q, d, verbose=verbose)

    ISD_min = [min(vals) for vals in zip(*ISD_l.values())]

    P = distribution_weight_folded_full(s, s_H, t, c, q, d, verbose)
    if len(P) < len(ISD_min):
        P = list(P) + [0.0] * (len(ISD_min) - len(P))
    else:
        P = list(P[: len(ISD_min)])

    Tmin = sum(C * p for C, p in zip(ISD_min, P))

    C_dict: dict[str, float] = {}
    for algo, vals in ISD_l.items():
        C_dict[algo] = sum(C * p for C, p in zip(vals, P))
    Tmin2 = min(C_dict.values())

    ratios: list[float] = []
    for a, b in zip(ISD_min, P):
        if b == 0:
            ratios.append(inf)
            continue
        denom = 1.0 - (1.0 - b) ** r
        if denom <= 0.0:
            ratios.append(inf)
            continue
        ratios.append((2.0 ** a + r * Cost_folding) / denom)
    ratio_min = min(ratios)
    w_min = ratios.index(ratio_min)
    Tmin3 = log2(ratio_min)

    if verbose:
        for algo, val in C_dict.items():
            print(f"{algo} --> {val}")
        print(f"Folded code has length {c * d ** fold_s}")
        print(f"Average running time of best ISD at each weight is {Tmin}")
        per_iter_log = log2(2.0 ** ISD_min[w_min] + Cost_folding)
        repeat_n = math.ceil(1 / P[w_min])
        repeat_log = math.ceil(log2(1 / P[w_min]))
        print(
            f"Weight which minimizes ISD/Proba is {w_min} which "
            f"happens with proba {P[w_min]}.\n\t--> Repeat "
            f"{repeat_n} (~=2^{repeat_log}) times "
            "ISD with aborts (each of them costing "
            f"2^{per_iter_log}) ---> {Tmin3}"
        )
        gb = _gaussian_binomial(s, s_H, d)
        print(
            f"There are {gb}~=2^{_log2_pos(gb)}  possible subgroups of "
            f"size {d}^{s_H}."
        )
    return min(Tmin, Tmin2, Tmin3)


def prepare_ISD_list(
    c: int,
    t: int,
    s: int,
    q: int,
    d: int | None = None,
    ISD_l: dict[str, list[float]] | None = None,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Cache ISD complexities up to weight c*t; extend in place if already partial."""
    if d is None:
        d = q - 1
    R = 1 - 1 / c
    gv = GV(R, q)
    fold_s = math.ceil(math.log(t / gv) / math.log(d))
    n = c * d ** fold_s
    k = (c - 1) * d ** fold_s

    if ISD_l is None:
        ISD_l = ISD_vec(c, t, fold_s, q, d, verbose=verbose)

    for name, ISD in ISDs.items():
        params: dict[str, Any] = {"p_min": 0, "ell_min": 0}
        ISD_complexity = ISD_l.get(name, [])
        for w in range(len(ISD_complexity), c * t + 1):
            res, params = ISD(w, k, n, q=q, verbose=verbose, full=True, **params)
            # NOTE: doom_loss is called with (d, fold_s) here, but with
            # (fold_s, d) in ISD_vec. The asymmetry is intentional.
            ISD_complexity.append(
                res + doom_loss(d, fold_s) + c_split_loss(n, c * t, c)
            )
        ISD_l[name] = ISD_complexity
    return ISD_l


def find_t(
    c: int,
    s: int,
    t: int | None = None,
    security_parameter: int = 128,
    q: int = 4,
    r: int = 1,
    d: int | None = None,
    verbose: bool = False,
) -> int:
    """Smallest t reaching the target security level for the given (c, s, q, d)."""
    if d is None:
        d = q - 1
    if not _is_prime_power(q):
        raise ValueError("Field size q must be a prime power.")
    if (q - 1) % d != 0:
        raise ValueError("d should divide q-1.")
    if t is None:
        t = refine_t(q, c, s)
    if verbose:
        print(f"Starting from t={t}.")
    ISD_l = prepare_ISD_list(c, t, s, q, d, ISD_l=None, verbose=verbose)
    cost = expect_cost(c, t, s, q, r, d, ISD_l, verbose)
    while cost < security_parameter:
        if verbose:
            print(f"\nt={t} --> {cost} bits. Too small. Testing t={t+1}.\n\n")
        t += 1
        ISD_l = prepare_ISD_list(c, t, s, q, d, ISD_l, verbose=verbose)
        cost = expect_cost(c, t, s, q, r, d, ISD_l, verbose=verbose)
    print(
        f"For s={s}, c={c}, q={q}, d={d}, we need t={t} for a "
        f"security estimated to {cost} bits"
    )
    return t
