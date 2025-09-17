import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

st.set_page_config(page_title="T‑tests, Correlation, and One‑way ANOVA", layout="wide")

st.title("T‑tests, Correlation, and One‑way ANOVA: First‑Year Sandbox")
st.write("Choose a simple design, simulate data, see the right test and plot, and read the output.")

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def cohen_d_ind(x, y):
    nx, ny = len(x), len(y)
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = ((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2)
    sp = np.sqrt(sp)
    return (np.mean(x) - np.mean(y)) / sp


def cohen_d_paired(x, y):
    d = x - y
    return np.mean(d) / np.std(d, ddof=1)


def eta_squared(aov_table):
    ss_effect = aov_table.loc[aov_table.index != 'Residual', 'sum_sq'].sum()
    ss_total = ss_effect + aov_table.loc['Residual', 'sum_sq']
    return ss_effect / ss_total


def simulate_independent_groups(n_per_group=30, levels=2, effect=0.6, noise_sd=1.0, seed=123):
    rng = np.random.default_rng(seed)
    means = [i * effect for i in range(levels)]
    rows = []
    for i in range(levels):
        y = rng.normal(means[i], noise_sd, size=n_per_group)
        for val in y:
            rows.append({"group": f"G{i+1}", "y": val, "subject": f"S{i+1}_{rng.integers(1, 1_000_000)}"})
    return pd.DataFrame(rows)


def simulate_paired(n_subjects=30, effect=0.5, noise_sd=1.0, rho=0.4, seed=123):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]]) * (noise_sd ** 2)
    base = rng.normal(0, 0.5, size=n_subjects)
    eps = rng.multivariate_normal([0, 0], cov, size=n_subjects)
    a = base + eps[:, 0]
    b = base + effect + eps[:, 1]
    rows = []
    for i in range(n_subjects):
        sid = f"S{i+1}"
        rows.append({"subject": sid, "cond": "A", "y": a[i]})
        rows.append({"subject": sid, "cond": "B", "y": b[i]})
    return pd.DataFrame(rows)


def simulate_correlation(n=80, slope=0.6, noise_sd=1.0, seed=123):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=n)
    y = 0.0 + slope * x + rng.normal(0, noise_sd, size=n)
    return pd.DataFrame({"x": x, "y": y})


def recommend_text(mode, levels):
    if mode == "Independent groups":
        if levels == 2:
            return "Independent samples t test. Equivalent to one way ANOVA with 2 levels."
        else:
            return "One way ANOVA."
    if mode == "Paired / repeated":
        return "Paired samples t test."
    return "Correlation and simple linear regression."

# -------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------

with st.sidebar:
    st.header("Design")
    mode = st.selectbox("Choose design", ["Independent groups", "Paired / repeated", "Correlation"], index=0)

    if mode == "Independent groups":
        levels = st.slider("Number of groups", 2, 6, 2, 1)
        n_per = st.slider("Participants per group", 5, 150, 30, 5)
        effect = st.slider("Group mean step (effect)", 0.0, 2.0, 0.6, 0.1)
        noise_sd = st.slider("Noise SD", 0.1, 3.0, 1.0, 0.1)
        error_type = st.radio("Error bars", ["SE", "SD"], index=0, horizontal=True)
        seed = st.number_input("Random seed", value=123, step=1)
    elif mode == "Paired / repeated":
        n_subj = st.slider("Participants", 5, 150, 30, 5)
        effect = st.slider("Mean difference B − A", 0.0, 2.0, 0.5, 0.1)
        rho = st.slider("Within person correlation", 0.0, 0.9, 0.4, 0.05)
        noise_sd = st.slider("Noise SD", 0.1, 3.0, 1.0, 0.1)
        error_type = st.radio("Error bars", ["SE", "SD"], index=0, horizontal=True)
        show_spaghetti = st.checkbox("Show subject lines", value=False)
        seed = st.number_input("Random seed", value=123, step=1)
    else:
        n = st.slider("Sample size", 20, 300, 80, 10)
        slope = st.slider("True slope", 0.0, 2.0, 0.6, 0.1)
        noise_sd = st.slider("Noise SD", 0.1, 3.0, 1.0, 0.1)
        seed = st.number_input("Random seed", value=123, step=1)

if mode == "Independent groups":
    st.info(recommend_text(mode, levels))
    df = simulate_independent_groups(n_per_group=n_per, levels=levels, effect=effect, noise_sd=noise_sd, seed=seed)
    st.subheader("Plot")
    agg = df.groupby("group")["y"].agg(["mean", "std", "count"]).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    err_col = "se" if error_type == "SE" else "std"
    fig = px.bar(agg, x="group", y="mean", error_y=err_col)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Analysis output")
    if levels == 2:
        g1 = df[df.group == "G1"]["y"].values
        g2 = df[df.group == "G2"]["y"].values
        t, p = stats.ttest_ind(g1, g2, equal_var=False)
        d = cohen_d_ind(g1, g2)
        st.write(f"Independent t test: t = {t:.3f}, p = {p:.4f}, Cohen d = {d:.3f}")
        model = smf.ols("y ~ C(group)", data=df).fit()
        aov = anova_lm(model, typ=2)
        st.caption("Equivalence: one way ANOVA on 2 groups gives the same inference as the t test.")
        st.dataframe(aov.round(4))
    else:
        model = smf.ols("y ~ C(group)", data=df).fit()
        aov = anova_lm(model, typ=2)
        st.dataframe(aov.round(4))
        try:
            etasq = eta_squared(aov)
            st.write(f"Eta squared = {etasq:.3f}")
        except Exception:
            pass

    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="independent_groups.csv", mime="text/csv")

elif mode == "Paired / repeated":
    st.info(recommend_text(mode, None))
    df = simulate_paired(n_subjects=n_subj, effect=effect, noise_sd=noise_sd, rho=rho, seed=seed)
    st.subheader("Plot")
    means = df.groupby("cond")["y"].agg(["mean", "std", "count"]).reset_index()
    means["se"] = means["std"] / np.sqrt(means["count"])
    err_col = "se" if error_type == "SE" else "std"
    fig = px.line(means, x="cond", y="mean", markers=True, error_y=err_col)
    if show_spaghetti:
        fig2 = px.line(df, x="cond", y="y", color="subject", line_group="subject", opacity=0.25)
        for tr in fig2.data:
            fig.add_trace(tr)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Analysis output")
    a = df[df.cond == "A"].sort_values("subject")["y"].values
    b = df[df.cond == "B"].sort_values("subject")["y"].values
    t, p = stats.ttest_rel(a, b)
    d = cohen_d_paired(a, b)
    st.write(f"Paired t test: t = {t:.3f}, p = {p:.4f}, Cohen d_z = {d:.3f}")

    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="paired.csv", mime="text/csv")

else:  # Correlation
    st.info(recommend_text(mode, None))
    df = simulate_correlation(n=n, slope=slope, noise_sd=noise_sd, seed=seed)
    st.subheader("Plot")
    fig = px.scatter(df, x="x", y="y", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Analysis output")
    r, p = stats.pearsonr(df["x"], df["y"])
    st.write(f"Pearson r = {r:.3f}, p = {p:.4f}")
    st.caption("Equivalence: correlation and simple linear regression test the same linear association.")

    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="correlation.csv", mime="text/csv")

st.markdown(
    """
Reading the outputs
- Independent groups with 2 levels uses an independent t test. With 3 or more levels use one way ANOVA.
- Paired design uses a paired t test on the same people across two conditions.
- Use SE or SD error bars from the sidebar. SE reflects certainty of the mean, SD reflects spread of scores.
- Correlation uses Pearson r, equivalent to the slope test in simple linear regression.
- Use the CSVs in JASP to practice reporting APA style results.
"""
)
