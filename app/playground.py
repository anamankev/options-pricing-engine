"""Playground: try the pricing engine with your own inputs and compare it with QuantLib.

    pip install -e ".[app]"          # once
    streamlit run app/playground.py  # opens http://localhost:8501 in your browser

Everything runs on your own computer; nothing is sent anywhere. The numbers come from
app/compare.py, which the test suite checks; this file only lays out the page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `app` and `experiments` import

from app.compare import (
    NOT_INSTALLED,
    TREE_METHODS,
    Contract,
    implied_vol_check,
    monte_carlo_check,
    monte_carlo_coverage,
    price_and_greeks,
    quantlib_installed,
    tree_convergence,
    tree_reference,
)
from pricing import black_scholes as bs
from pricing.contracts import ExerciseStyle, OptionType
from pricing.errors import PricingInputError
from pricing.monte_carlo import MCMethod

# Study B's figure colours (experiments/study_b_trees.py), so the methods look the same everywhere.
COLOR = {"CRR": "#2a78d6", "Leisen-Reimer": "#eb6834", "BBS": "#1baf7a", "BBSR": "#eda100"}

st.set_page_config(page_title="Options pricing playground", layout="wide")
st.title("Options pricing playground")
st.caption(
    "Type in an option. Each tab shows this library's answer beside an independent one "
    "(QuantLib, the exact formula, or the true volatility) and whether they agree within the "
    "same tolerance the tests use."
)
if not quantlib_installed():
    st.warning(f"Comparison columns are empty because {NOT_INSTALLED}.")

with st.sidebar:
    st.header("The option")
    kind = st.radio("Type", ["Call", "Put"], horizontal=True)
    S = st.number_input("Spot price S", value=42.0, min_value=0.01, format="%.4f")
    K = st.number_input("Strike K", value=40.0, min_value=0.01, format="%.4f")
    days = st.number_input("Days to expiry", value=180, min_value=1, step=1)
    year_days = st.radio("Days per year", [360, 365], horizontal=True)
    r = st.number_input("Interest rate r (0.05 = 5%)", value=0.10, format="%.4f")
    q = st.number_input("Dividend yield q", value=0.0, format="%.4f")
    sigma = st.number_input(
        "Volatility sigma (0.20 = 20%)", value=0.20, min_value=0.0001, format="%.4f"
    )
    st.caption(
        "Defaults: Hull, Example 15.6: 180 days on a 360-day year is T = 0.5; call price 4.76."
    )

contract = Contract(
    OptionType.CALL if kind == "Call" else OptionType.PUT, S, K, int(days), r, q, sigma, year_days
)
tab_price, tab_trees, tab_mc, tab_iv = st.tabs(
    ["Price and Greeks", "Trees (American)", "Monte Carlo", "Implied volatility"]
)


def verdict(agrees: bool | None) -> str:
    return "-" if agrees is None else ("PASS" if agrees else "FAIL")


with tab_price:
    st.subheader("European price and Greeks: this library vs QuantLib")
    try:
        rows = price_and_greeks(contract)
        st.dataframe(
            [
                {
                    "Quantity": row.quantity,
                    "This library": f"{row.ours:.12g}",
                    "QuantLib": "-" if row.quantlib is None else f"{row.quantlib:.12g}",
                    "Difference": "-" if row.difference is None else f"{row.difference:.2e}",
                    "Agree?": verdict(row.agrees),
                }
                for row in rows
            ],
            hide_index=True,
        )
        st.caption(
            "Units (SPEC 2): Greeks per 1.00 change in the input; vega and rho per 1.00 of sigma "
            "and r; theta is the change per year. Agree = within 1e-11 relative or 4 eps x the "
            "quantity's scale."
        )
    except PricingInputError as exc:
        st.error(f"Invalid input: {exc}")

with tab_trees:
    st.subheader("Binomial trees as the number of steps N grows")
    exercise_name = st.radio("Exercise", ["American", "European"], horizontal=True)
    exercise = ExerciseStyle.AMERICAN if exercise_name == "American" else ExerciseStyle.EUROPEAN
    n_max = st.select_slider("Largest N", options=[100, 200, 400, 800, 1600], value=400)
    steps = list(range(10, n_max + 1, max(1, n_max // 60)))
    try:
        reference, source = tree_reference(contract, exercise)
        values = tree_convergence(contract, exercise, steps)
        fig, ax = plt.subplots(figsize=(8, 4))
        for method in TREE_METHODS:
            ax.plot(steps, values[method], label=method, color=COLOR[method], linewidth=1.5)
        if reference is not None:
            ax.axhline(reference, color="#52514e", linestyle="--", linewidth=1, label="reference")
        ax.set_xlabel("steps N")
        ax.set_ylabel("price")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.write(f"Reference: {source}" + ("" if reference is None else f" = {reference:.10f}"))
        last = {m: values[m][-1] for m in TREE_METHODS}
        st.dataframe(
            [
                {
                    "Method": m,
                    f"Price at N = {steps[-1]}": f"{v:.10f}",
                    "Error": "-" if reference is None else f"{v - reference:+.2e}",
                }
                for m, v in last.items()
            ],
            hide_index=True,
        )
        st.caption("CRR zigzags between odd and even N; Leisen-Reimer and BBSR converge faster.")
    except PricingInputError as exc:
        st.error(f"Invalid input: {exc}")

with tab_mc:
    st.subheader("Monte Carlo against the exact European price")
    method_name = st.radio("Method", ["plain", "antithetic", "control variate"], horizontal=True)
    n_paths = st.select_slider("Paths", options=[1_000, 10_000, 100_000, 1_000_000], value=100_000)
    seed = st.number_input("Seed (same seed, same answer)", value=1, min_value=0, step=1)
    method = {
        "plain": MCMethod.PLAIN,
        "antithetic": MCMethod.ANTITHETIC,
        "control variate": MCMethod.CONTROL_VARIATE,
    }[method_name]
    try:
        mc = monte_carlo_check(contract, int(n_paths), int(seed), method)
        cols = st.columns(3)
        cols[0].metric(
            "Monte Carlo estimate",
            f"{mc.estimate:.6f}",
            f"SE {mc.std_error:.2e}",
            delta_color="off",
        )
        cols[1].metric("Exact price", f"{mc.exact:.6f}")
        cols[2].metric("95% interval covers it?", "yes" if mc.covered else "no")
        st.write(f"95% interval: [{mc.ci_low:.6f}, {mc.ci_high:.6f}]")
        if not mc.covered:
            st.info(
                "This seed's interval misses the exact price. A 95% interval should miss about "
                "1 time in 20, so one miss proves nothing: check many seeds below."
            )
        seeds = st.select_slider("Check over many seeds", options=[100, 200, 500], value=200)
        if st.button(f"Run {seeds} seeds"):
            share = monte_carlo_coverage(contract, int(n_paths), method, int(seeds))
            band = 4 * (0.95 * 0.05 / seeds) ** 0.5
            ok = abs(share - 0.95) <= band
            st.metric(
                f"Coverage over {seeds} seeds", f"{share:.3f}", "expected 0.95", delta_color="off"
            )
            (st.success if ok else st.error)(
                f"{'Within' if ok else 'Outside'} 0.95 +/- {band:.3f} (4 standard errors)."
            )
    except PricingInputError as exc:
        st.error(f"Invalid input: {exc}")

with tab_iv:
    st.subheader("Implied volatility: price in, volatility out")
    st.write(
        "By default the price is computed from your sigma, so every solver should give it back."
    )
    try:
        model_price = float(bs.price(contract.option_type, **contract.market()))
        price = st.number_input("Option price", value=model_price, format="%.10f")
        known = sigma if price == model_price else None
        rows = implied_vol_check(contract, price, known)
        st.dataframe(
            [
                {
                    "Solver": row.solver,
                    "Implied sigma": "-" if row.sigma is None else f"{row.sigma:.15g}",
                    "Iterations": "-" if row.iterations is None else row.iterations,
                    "Stop reason": row.stop_reason,
                    "Error vs your sigma": "-" if row.error is None else f"{row.error:.1e}",
                }
                for row in rows
            ],
            hide_index=True,
        )
        st.caption(
            "A solver returns a volatility only when it converged; otherwise it says why it "
            "stopped. at_lower_bound means the price carries no information about volatility "
            "(for example, below the rounding of the intrinsic value)."
        )
    except PricingInputError as exc:
        st.error(f"Invalid input: {exc}")
