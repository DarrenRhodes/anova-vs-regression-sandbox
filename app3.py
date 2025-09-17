import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

st.set_page_config(page_title="T-tests, Correlation, and One-way ANOVA", layout="wide")

st.title("T-tests, Correlation, and One-way ANOVA: First-Year Sandbox")
st.write("Choose a simple design, simulate data, see the right test and plot, and read the output.")

# -----------------------------
# Helpers
# -----------------------------
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
    ss_effect = aov_table.loc[aov_table.index != "Residual", "sum_sq"].sum()
    ss_total = ss_effect + aov_table.loc["Residual", "sum_sq"]
    return float(ss_effect / ss_total)

def aggregate_stats(df, key_col, ycol="y"):
    agg = df.groupby(key_col)[ycol].agg(["mean", "std", "count"]).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    return agg

def simulate_independent_groups(n_per_group=30, levels=2, effect=0.6, noise_sd=1.0, seed=123):
    rng = np.random.default_rng(seed)
    means = [i * effect for i in range(levels)]
    rows = []
    for i in range(levels):
        y = rng.normal(means[i], noise_sd, size=n_per_group)
        for val in y:
            rows.append({"group": f"G{i+1}", "y": float(val), "subject": f"S{i+1}_{rng.integers(1, 1_000_000)}"})
    return pd.DataFrame(rows)

def simulate_within_levels(n_subjects=30, levels=2, step=0.5, noise_sd=1.0, rho=0.4, seed=123):
    rng = np.random.default_rng(seed)
    means = np.array([i * step for i in range(levels)])
    cov = np.full((levels, levels), rho)
    np.fill_diagonal(cov, 1.0)
    cov = cov * (noise_sd ** 2)
    rows = []
    for s in range(n_subjects):
        base = rng.normal(0, 0.5)
        eps = rng.multivariate_normal(np.zeros(levels), cov)
        for i in range(levels):
            y = base + means[i] + eps[i]
            rows.append({"subject": f"S{s+1}", "cond": f"C{i+1}", "y": float(y)})
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
        if levels == 2:
            return "Paired samples t test."
        else:
            return "One way repeated measures ANOVA."
    return "Correlation and simple linear regression."

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Design")
    mode = st.selectbox("Choose design", ["Independent groups", "Paired / repeated", "Correlation"], index=0)

    if mode == "Independent groups":
        levels = st.slider("Number of groups", 2, 6, 3, 1)
        n_per = st.slider("Participants per group", 5, 150, 30, 5)
        effect = st.slider("Group mean step", 0.0, 2.0, 0.6, 0.1)
        noise_sd = st.slider("Noise SD", 0.1, 3.0, 1.0, 0.1)
        error_type = st.radio("Error bars", ["SE", "SD"], index=0, horizontal=True)
        show_points = st.checkbox("Overlay individual points", value=True)
        point_style = st.radio("Point placement", ["Overlay", "Side"], index=0, horizontal=True)
        seed = st.number_input("Random seed", value=123, step=1)
    elif mode == "Paired / repeated":
        levels = st.slider("Number of conditions", 2, 6, 2, 1)
        n_subj = st.slider("Participants", 5, 150, 30, 5)
        step = st.slider("Condition mean step", 0.0, 2.0, 0.5, 0.1)
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

# -----------------------------
# Main logic
# -----------------------------
if mode == "Independent groups":
    st.info(recommend_text(mode, levels))
    df = simulate_independent_groups(n_per_group=n_per, levels=levels, effect=effect, noise_sd=noise_sd, seed=seed)

    st.subheader("Plot")
    agg = aggregate_stats(df, "group")
    err_col = "se" if error_type == "SE" else "std"

    # Use numeric x for both bars and points so jitter works for every group
    labels = agg["group"].tolist()
    x_pos = np.arange(len(labels))

    fig = go.Figure()

    # Bars with error bars
    fig.add_trace(go.Bar(
        x=x_pos,
        y=agg["mean"],
        error_y=dict(type="data", array=agg[err_col], visible=True),
        name="Means",
        hovertemplate="Group=%{customdata}<br>Mean=%{y:.3f}<extra></extra>",
        customdata=labels
    ))

    # Points overlaid or to the side, jittered per group
    if show_points:
        jitter_scale = 0.08 if point_style == "Overlay" else 0.2
        side_shift = 0.0 if point_style == "Overlay" else 0.25
        for i, g in enumerate(labels):
            gdf = df[df["group"] == g]
            rng = np.random.default_rng(seed + i)
            x_jit = i + side_shift + rng.normal(0, jitter_scale, size=len(gdf))
            fig.add_trace(go.Scatter(
                x=x_jit,
                y=gdf["y"],
                mode="markers",
                name=f"Points {g}",
                marker=dict(size=6, opacity=0.55),
                hovertemplate=f"Group={g}<br>y=%{{y:.3f}}<extra></extra>",
                showlegend=True
            ))

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(
            tickmode="array",
            tickvals=x_pos,
            ticktext=labels,
            title="Group"
        ),
        yaxis=dict(title="DV")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Analysis output")
    if levels == 2:
        g1 = df[df.group == "G1"].sort_values("subject")["y"].values
        g2 = df[df.group == "G2"].sort_values("subject")["y"].values
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

    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="independent_groups.csv", mime="text/csv")

elif mode == "Paired / repeated":
    st.info(recommend_text(mode, levels))
    df = simulate_within_levels(n_subjects=n_subj, levels=levels, step=step, noise_sd=noise_sd, rho=rho, seed=seed)

    st.subheader("Plot")
    means = aggregate_stats(df, "cond")
    err_col = "se" if error_type == "SE" else "std"

    # Order C1, C2, ...
    cond_order = sorted(means["cond"].unique(), key=lambda c: int(c[1:]))
    means = means.set_index("cond").reindex(cond_order).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=means["cond"],
        y=means["mean"],
        mode="lines+markers",
        name="Means",
        error_y=dict(type="data", array=means[err_col], visible=True),
        hovertemplate="Cond=%{x}<br>Mean=%{y:.3f}<extra></extra>"
    ))

    if show_spaghetti:
        for sid, subdf in df.groupby("subject"):
            subdf = subdf.set_index("cond").reindex(cond_order).reset_index()
            fig.add_trace(go.Scatter(
                x=subdf["cond"],
                y=subdf["y"],
                mode="lines",
                line=dict(width=1),
                name=str(sid),
                showlegend=False,
                opacity=0.15,
                hovertemplate=f"Subject={sid}<br>Cond=%{{x}}<br>y=%{{y:.3f}}<extra></extra>"
            ))

    fig.update_layout(xaxis_title="Condition", yaxis_title="DV")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Analysis output")
    if levels == 2:
        a = df[df.cond == "C1"].sort_values("subject")["y"].values
        b = df[df.cond == "C2"].sort_values("subject")["y"].values
        t, p = stats.ttest_rel(a, b)
        d = cohen_d_paired(a, b)
        st.write(f"Paired t test: t = {t:.3f}, p = {p:.4f}, Cohen d_z = {d:.3f}")
    else:
        aovrm = sm.stats.AnovaRM(df, depvar="y", subject="subject", within=["cond"]).fit()
        st.dataframe(aovrm.anova_table.round(4))

    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="paired_rm.csv", mime="text/csv")

else:
    st.info(recommend_text(mode, None))
    df = simulate_correlation(n=n, slope=slope, noise_sd=noise_sd, seed=seed)

    st.subheader("Plot")
    slope_hat, intercept_hat, r_val, p_val, _ = stats.linregress(df["x"], df["y"])
    xline = np.linspace(df["x"].min(), df["x"].max(), 100)
    yline = intercept_hat + slope_hat * xline

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["x"], y=df["y"],
        mode="markers",
        name="Data",
        marker=dict(size=6, opacity=0.7),
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=xline, y=yline,
        mode="lines",
        name="OLS fit",
        hovertemplate="y = a + b x<extra></extra>"
    ))
    fig.update_layout(xaxis_title="x", yaxis_title="y")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Analysis output")
    r, p = stats.pearsonr(df["x"], df["y"])
    st.write(f"Pearson r = {r:.3f}, p = {p:.4f}")
    st.caption("Equivalence: correlation and simple linear regression test the same linear association.")

    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="correlation.csv", mime="text/csv")

st.markdown(
    """
Reading the outputs
- Independent groups with 2 levels uses an independent t test. With 3 or more levels use one way ANOVA.
- Paired or repeated designs: 2 conditions use a paired t test. 3 or more conditions use one way repeated measures ANOVA.
- Use SE or SD error bars from the sidebar. SE reflects certainty of the mean. SD reflects spread of scores.
- Overlay points or subject lines to see variability.
- Correlation uses Pearson r, equivalent to the slope test in simple linear regression.
- Use the CSVs in JASP to practice reporting APA style results.
"""
)
