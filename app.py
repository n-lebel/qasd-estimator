"""Streamlit UI for the QA-SD security estimator.

Calls `estimator.expect_cost` and `estimator.find_t` directly. No Sage
subprocess required. Run with `streamlit run app.py` from this directory.
"""
import contextlib
import io
from typing import Any, Callable

import streamlit as st

import estimator


def _run_with_capture(fn: Callable[..., Any], **kwargs: Any) -> tuple[Any, str, str | None]:
    """Run fn(**kwargs), capture stdout, return (result, stdout, error_str)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = fn(**kwargs)
        return result, buf.getvalue(), None
    except Exception as exc:  # noqa: BLE001 — surface any estimator failure to the UI
        return None, buf.getvalue(), f"{type(exc).__name__}: {exc}"


def render_output(stdout: str, error: str | None) -> None:
    if error:
        st.error(error)
    if stdout:
        with st.expander("Verbose output"):
            st.code(stdout)


def expect_cost_form() -> None:
    with st.form("expect_cost_form"):
        col1, col2 = st.columns(2)
        c = col1.number_input("c (compression factor)", min_value=2, value=5, step=1)
        t = col2.number_input("t (errors per block)", min_value=1, value=14, step=1)
        s = col1.number_input("s (log group size)", min_value=1, value=16, step=1)
        q_bits = col2.number_input(
            "q_bits (bit size of prime field, uses q = 2^q_bits − 1)",
            min_value=2,
            value=107,
            step=1,
        )

        with st.expander("Advanced"):
            r = st.number_input("r (SSD repetitions)", min_value=1, value=1, step=1)
            d_str = st.text_input("d (group degree, blank = 2)", value="")
            offset = st.number_input("offset (folding offset)", value=0, step=1)
            verbose = st.checkbox("Verbose output", value=True)
        submit = st.form_submit_button("Run")

    if not submit:
        return

    kwargs: dict[str, Any] = {
        "c": int(c),
        "t": int(t),
        "s": int(s),
        "q_bits": int(q_bits),
        "r": int(r),
        "offset": int(offset),
        "verbose": verbose,
    }
    if d_str.strip():
        try:
            kwargs["d"] = int(d_str)
        except ValueError:
            st.error(f"Invalid d: {d_str!r}")
            return

    with st.spinner("Running estimator…"):
        result, stdout, error = _run_with_capture(estimator.expect_cost, **kwargs)

    if error is None and result is not None:
        m1, m2 = st.columns(2)
        m1.metric("Bits of security", f"{result:.4f}")
        m2.metric("Complexity (individual DPF calls)", f"{(int(c) * int(t)) ** 2:,}")
    render_output(stdout, error)


def find_t_form() -> None:
    with st.form("find_t_form"):
        col1, col2 = st.columns(2)
        c = col1.number_input("c (compression factor)", min_value=2, value=5, step=1)
        s = col2.number_input("s (log group size)", min_value=1, value=15, step=1)
        q_bits = col1.number_input(
            "q_bits (bit size of prime field, uses q = 2^q_bits − 1)",
            min_value=2,
            value=107,
            step=1,
        )
        security_parameter = col2.number_input(
            "Target security (bits)", min_value=1, value=128, step=1
        )

        with st.expander("Advanced"):
            t_start = st.text_input("t (starting lower bound, blank = auto)", value="")
            r = st.number_input("r (SSD repetitions)", min_value=1, value=1, step=1)
            d_str = st.text_input("d (group degree, blank = 2)", value="")
            verbose = st.checkbox("Verbose output", value=True)
        submit = st.form_submit_button("Run")

    if not submit:
        return

    kwargs: dict[str, Any] = {
        "c": int(c),
        "s": int(s),
        "q_bits": int(q_bits),
        "security_parameter": int(security_parameter),
        "r": int(r),
        "verbose": verbose,
    }
    if t_start.strip():
        try:
            kwargs["t"] = int(t_start)
        except ValueError:
            st.error(f"Invalid t: {t_start!r}")
            return
    if d_str.strip():
        try:
            kwargs["d"] = int(d_str)
        except ValueError:
            st.error(f"Invalid d: {d_str!r}")
            return

    with st.spinner("Running estimator…"):
        t_result, stdout, error = _run_with_capture(estimator.find_t, **kwargs)
    if error is not None or t_result is None:
        render_output(stdout, error)
        return

    expect_kwargs: dict[str, Any] = {
        "c": kwargs["c"],
        "t": int(t_result),
        "s": kwargs["s"],
        "q_bits": kwargs["q_bits"],
        "r": kwargs.get("r", 1),
    }
    if "d" in kwargs:
        expect_kwargs["d"] = kwargs["d"]
    bits, bits_stdout, bits_error = _run_with_capture(
        estimator.expect_cost, **expect_kwargs
    )
    complexity = (int(kwargs["c"]) * int(t_result)) ** 2
    m1, m2 = st.columns(2)
    if bits_error is None and bits is not None:
        m1.metric("Required t", f"{int(t_result)} ({bits:.2f} bits)")
    else:
        m1.metric("Required t", f"{int(t_result)}")
    m2.metric("Complexity (individual DPF calls)", f"{complexity:,}")
    render_output(stdout + bits_stdout, bits_error)


def main() -> None:
    st.set_page_config(page_title="QA-SD security estimator", layout="centered")
    st.title("QA-SD security estimator")
    st.caption(
        "Security estimator for parameter sets of the quasi-abelian syndrome decoding problem and its stationary variant.\n"
        "Adapted by the authors of [[LLXY+26]](https://eprint.iacr.org/2026/196.pdf) to include the newer attack from [[BDHV25]](https://eprint.iacr.org/2025/892.pdf).\n"
        "Original code: [FOLEAGE PCG estimator](https://github.com/mbombar/estimator_folding). All credits to the aforementioned contributors.\n"
    )
    noise = st.radio(
        "Noise distribution",
        ["Regular", "Uniform"],
        horizontal=True,
        help=(
            "Regular: the error vector is split into c blocks of length d^s, each with "
            "exactly t errors. The estimator adds a provable lower-bound penalty for "
            "putative attacks exploiting the block structure (csplit=True). "
            "Uniform: the error has weight c·t spread arbitrarily over the whole "
            "code, with no block structure (csplit=False)."
        ),
    )
    estimator.set_aggressivity(csplit_=(noise == "Regular"))
    mode = st.radio(
        "Mode",
        ["Estimate cost (expect_cost)", "Find required t (find_t)"],
        horizontal=True,
    )
    if mode.startswith("Estimate"):
        expect_cost_form()
    else:
        find_t_form()


main()
