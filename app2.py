import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(page_title="Within vs Between Sandbox", layout="wide")

st.title("Within vs Between: Repeated Measures and Mixed Designs")
st.write("Explore how design structure changes the right analysis and the right plot.")

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def make_effects(levels, base=0.0, step=0.6):
    return np.array([base + i * step for i in range(levels)])


def corr_cov_matrix(k, rho, sd):
    M = np.full((k, k), rho)
    np.fill_diagonal(M, 1.0)
    return (sd ** 2) * M


def simulate_between_only(n_per_group, levels_A, sd_noise, sd_subject, seed=123):
    rng = np.random.default_rng(seed)
    A_eff = make_effects(levels_A, step=0.6)
    rows = []
    for i in range(levels_A):
        for s in range(n_per_group):
            subject = f"S_A{i+1}_{s+1}"
            subj_re = rng.normal(0, sd_subject)
            y = A_eff[i] + subj_re + rng.normal(0, sd_noise)
            rows.append({"subject": subject, "A": f"A{i+1}", "y": y})
    return pd.DataFrame(rows)


def simulate_within_only(n_subjects, levels_A, sd_noise, sd_subject, rho, seed=123):
    rng = np.random.default_rng(seed)
    A_eff = make_effects(levels_A, step=0.6)
    cov = corr_cov_matrix(levels_A, rho, sd_noise)
    rows = []
    for s in range(n_subjects):
        subject = f"S{s+1}"
        subj_re = rng.normal(0, sd_subject)
        eps = rng.multivariate_normal(mean=np.zeros(levels_A), cov=cov)
        for i in range(levels_A):
            y = A_eff[i] + subj_re + eps[i]
            rows.append({"subject": subject, "A": f"A{i+1}", "y": y})
    return pd.DataFrame(rows)


def simulate_two_between(n_per_cell, levels_A, levels_B, sd_noise, sd_subject, seed=123, interaction=0.0):
    rng = np.random.default_rng(seed)
    A_eff = make_effects(levels_A, step=0.6)
    B_eff = make_effects(levels_B, step=0.4)
    rows = []
    for i in range(levels_A):
        for j in range(levels_B):
            mu = A_eff[i] + B_eff[j] + interaction * i * j
            for s in range(n_per_cell):
                subject = f"S_A{i+1}_B{j+1}_{s+1}"
                subj_re = rng.normal(0, sd_subject)
                y = mu + subj_re + rng.normal(0, sd_noise)
                rows.append({"subject": subject, "A": f"A{i+1}", "B": f"B{j+1}", "y": y})
    return pd.DataFrame(rows)


def simulate_two_within(n_subjects, levels_A, levels_B, sd_noise, sd_subject, rho_A, rho_B, seed=123, interaction=0.0):
    # Use independent AR1 style correlations across A and B by constructing covariance on the Kronecker product grid
    rng = np.random.default_rng(seed)
    A_eff = make_effects(levels_A, step=0.6)
    B_eff = make_effects(levels_B, step=0.4)
    # Build covariance using separable structure
    CA = corr_cov_matrix(levels_A, rho_A, 1.0)
    CB = corr_cov_matrix(levels_B, rho_B, 1.0)
    C = np.kron(CA, CB) * (sd_noise ** 2)
    rows = []
    for s in range(n_subjects):
        subject = f"S{s+1}"
        subj_re = rng.normal(0, sd_subject)
        eps = rng.multivariate_normal(mean=np.zeros(levels_A * levels_B), cov=C)
        k = 0
        for i in range(levels_A):
            for j in range(levels_B):
                mu = A_eff[i] + B_eff[j] + interaction * i * j
                y = mu + subj_re + eps[k]
                rows.append({"subject": subject, "A": f"A{i+1}", "B": f"B{j+1}", "y": y})
                k += 1
    return pd.DataFrame(rows)


def simulate_mixed(n_per_group, levels_within, levels_between, within_name, between_name,
                    sd_noise, sd_subject, rho, seed=123, interaction=0.0):
    # within_name is "A" or "B"
    rng = np.random.default_rng(seed)
    W_eff = make_effects(levels_within, step=0.6)
    B_eff = make_effects(levels_between, step=0.4)
    cov = corr_cov_matrix(levels_within, rho, sd_noise)
    rows = []
    for j in range(levels_between):
        for s in range(n_per_group):
            subject = f"S_{between_name}{j+1}_{s+1}"
            subj_re = rng.normal(0, sd_subject)
            eps = rng.multivariate_normal(mean=np.zeros(levels_within), cov=cov)
            for i in range(levels_within):
                # map to A, B labels regardless of which is within
                if within_name == "A":
                    A_lbl = f"A{i+1}"
                    B_lbl = f"B{j+1}"
                    mu = W_eff[i] + B_eff[j] + interaction * i * j
                else:
                    A_lbl = f"A{j+1}"
                    B_lbl = f"B{i+1}"
                    mu = W_eff[i] + B_eff[j] + interaction * j * i
                y = mu + subj_re + eps[i]
                rows.append({"subject": subject, "A": A_lbl, "B": B_lbl, "y": y})
    return pd.DataFrame(rows)


def recommend(dstruct, factors):
    if dstruct == "One factor":
        typ = factors[0][1]
        if typ == "Between":
            return "One way between subjects ANOVA"
        else:
            return "One way repeated measures ANOVA"
    else:
        types = [t for _, t in factors]
        if types == ["Between", "Between"]:
            return "Two way between subjects ANOVA"
        if types == ["Within", "Within"]:
            return "Two way repeated measures ANOVA"
        return "Mixed ANOVA via linear mixed effects"


def run_analysis(df, dstruct, factors):
    # factors is list of tuples [("A", type), ("B", type?)]
    out = {}
    if dstruct == "One factor":
        Atype = factors[0][1]
        if Atype == "Between":
            model = smf.ols("y ~ C(A)", data=df).fit()
            out["anova"] = anova_lm(model, typ=2)
        else:
            # Repeated measures with AnovaRM
            aovrm = sm.stats.AnovaRM(df, "y", "subject", within=["A"]).fit()
            out["aovrm"] = aovrm.anova_table
        return out

    # Two factors
    Atype, Btype = factors[0][1], factors[1][1]
    if Atype == "Between" and Btype == "Between":
        model = smf.ols("y ~ C(A) * C(B)", data=df).fit()
        out["anova"] = anova_lm(model, typ=2)
        return out

    if Atype == "Within" and Btype == "Within":
        aovrm = sm.stats.AnovaRM(df, "y", "subject", within=["A", "B"]).fit()
        out["aovrm"] = aovrm.anova_table
        return out

    # Mixed case: one within, one between
    # Use MixedLM with random intercepts by subject
    model = smf.mixedlm("y ~ C(A) * C(B)", data=df, groups=df["subject"]).fit(method="lbfgs", reml=True)
    out["mixedlm_summary"] = model.summary().as_text()
    return out


def agg_means_se(df, keys):
    g = df.groupby(keys)["y"].agg(["mean", "std", "count"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"])
    return g


def plot_data(df, dstruct, factors, spaghetti=False):
    if dstruct == "One factor":
        Atype = factors[0][1]
        if Atype == "Between":
            agg = agg_means_se(df, ["A"])
            fig = px.bar(agg, x="A", y="mean", error_y="se")
        else:
            agg = agg_means_se(df, ["A"])
            fig = px.line(agg, x="A", y="mean", markers=True)
            if spaghetti:
                fig2 = px.line(df, x="A", y="y", color="subject", opacity=0.25)
                for d in fig2.data:
                    fig.add_trace(d)
        return fig

    # Two factor plots
    Atype, Btype = factors[0][1], factors[1][1]
    if Atype == "Between" and Btype == "Between":
        agg = agg_means_se(df, ["A", "B"])
        fig = px.bar(agg, x="A", y="mean", color="B", barmode="group", error_y="se")
        return fig

    if Atype == "Within" and Btype == "Within":
        agg = agg_means_se(df, ["A", "B"])
        fig = px.line(agg, x="A", y="mean", color="B", markers=True)
        if spaghetti:
            fig2 = px.line(df, x="A", y="y", color="subject", line_group="subject", facet_col="B", facet_col_wrap=0, opacity=0.2)
            for d in fig2.data:
                fig.add_trace(d)
        return fig

    # Mixed: show within factor on x axis, color by between factor
    # Determine which is within
    if Atype == "Within" and Btype == "Between":
        agg = agg_means_se(df, ["A", "B"])
        fig = px.line(agg, x="A", y="mean", color="B", markers=True)
        if spaghetti:
            fig2 = px.line(df, x="A", y="y", color="subject", line_group="subject", facet_col="B", opacity=0.2)
            for d in fig2.data:
                fig.add_trace(d)
        return fig
    else:
        # A between, B within
        agg = agg_means_se(df, ["B", "A"])
        fig = px.line(agg, x="B", y="mean", color="A", markers=True)
        if spaghetti:
            fig2 = px.line(df, x="B", y="y", color="subject", line_group="subject", facet_col="A", opacity=0.2)
            for d in fig2.data:
                fig.add_trace(d)
        return fig

# -------------------------------------------------------------
# UI
# -------------------------------------------------------------

with st.sidebar:
    st.header("Design")
    design = st.selectbox("How many factors", ["One factor", "Two factors"], index=0)

    if design == "One factor":
        levels_A = st.slider("Levels for factor A", 2, 6, 3, 1)
        type_A = st.selectbox("A is", ["Between", "Within"], index=0)
    else:
        levels_A = st.slider("Levels for factor A", 2, 6, 2, 1)
        levels_B = st.slider("Levels for factor B", 2, 6, 2, 1)
        type_A = st.selectbox("A is", ["Between", "Within"], index=0)
        type_B = st.selectbox("B is", ["Between", "Within"], index=1)

    st.subheader("Sampling")
    if design == "One factor":
        if type_A == "Between":
            n_per_group = st.slider("Participants per level", 5, 100, 30, 5)
        else:
            n_per_group = st.slider("Participants", 5, 100, 30, 5)
    else:
        if type_A == "Between" and type_B == "Between":
            n_per_group = st.slider("Participants per cell", 4, 80, 20, 4)
        elif type_A == "Within" and type_B == "Within":
            n_per_group = st.slider("Participants", 5, 100, 30, 5)
        else:
            n_per_group = st.slider("Participants per between level", 4, 80, 15, 1)

    st.subheader("Data properties")
    sd_noise = st.slider("Residual SD", 0.1, 3.0, 1.0, 0.1)
    sd_subject = st.slider("Subject SD", 0.0, 2.0, 0.5, 0.1)
    rho = st.slider("Within measures correlation", 0.0, 0.9, 0.4, 0.05)
    interaction = st.slider("Interaction strength", 0.0, 2.0, 0.3, 0.1)
    seed = st.number_input("Random seed", value=123, step=1)
    spaghetti = st.checkbox("Show subject trajectories", value=False)

factors = [("A", type_A)] if design == "One factor" else [("A", type_A), ("B", type_B)]
st.info(f"Recommended analysis: {recommend(design, factors)}")

# Simulate according to design
if design == "One factor":
    if type_A == "Between":
        df = simulate_between_only(n_per_group, levels_A, sd_noise, sd_subject, seed)
    else:
        df = simulate_within_only(n_per_group, levels_A, sd_noise, sd_subject, rho, seed)
else:
    if type_A == "Between" and type_B == "Between":
        df = simulate_two_between(n_per_group, levels_A, levels_B, sd_noise, sd_subject, seed, interaction)
    elif type_A == "Within" and type_B == "Within":
        df = simulate_two_within(n_per_group, levels_A, levels_B, sd_noise, sd_subject, rho, rho, seed, interaction)
    elif type_A == "Within" and type_B == "Between":
        df = simulate_mixed(n_per_group, levels_A, levels_B, within_name="A", between_name="B",
                            sd_noise=sd_noise, sd_subject=sd_subject, rho=rho, seed=seed, interaction=interaction)
    else:
        df = simulate_mixed(n_per_group, levels_B, levels_A, within_name="B", between_name="A",
                            sd_noise=sd_noise, sd_subject=sd_subject, rho=rho, seed=seed, interaction=interaction)

# Plot
st.subheader("Plot")
fig = plot_data(df, design, factors, spaghetti=spaghetti)
st.plotly_chart(fig, use_container_width=True)

# Analysis
st.subheader("Analysis output")
res = run_analysis(df, design, factors)
if "anova" in res:
    st.dataframe(res["anova"].round(4))
if "aovrm" in res:
    st.dataframe(res["aovrm"].round(4))
if "mixedlm_summary" in res:
    st.text(res["mixedlm_summary"])  # show fixed effect table and variance components

# Download
st.download_button(
    label="Download simulated data (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="simulated_data_within_between.csv",
    mime="text/csv",
)

st.markdown(
    """
Notes for interpretation
- Between designs: bars with error bars. OLS ANOVA table shown.
- Within designs: lines for condition means. Repeated measures ANOVA table shown.
- Mixed designs: lines by the within factor, colored by the between factor. Mixed model summary shown. Use fixed effects for tests.
"""
)
