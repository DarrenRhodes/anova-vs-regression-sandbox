import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

st.set_page_config(page_title="ANOVA vs Regression Sandbox", layout="wide")

st.title("ANOVA vs Regression: Interactive Sandbox")
st.write(
    "Play with variable types, generate data, and see the matching analysis and plot."
)

# -----------------------------
# Helpers
# -----------------------------

def recommend_analysis(dv_type, x1_type, x2_type=None):
    if dv_type == "Continuous":
        if x2_type is None:
            if x1_type == "Categorical":
                return "One-way ANOVA or t-test (if 2 levels). Equivalent to regression with dummy coding."
            else:
                return "Simple linear regression."
        else:
            if x1_type == "Categorical" and x2_type == "Categorical":
                return "Two-way ANOVA. Equivalent to multiple regression with interaction and dummies."
            if x1_type == "Continuous" and x2_type == "Continuous":
                return "Multiple linear regression with interaction optional."
            else:
                return "ANCOVA or linear regression with one dummy-coded factor and one continuous predictor."
    else:  # Binary DV
        return "Logistic regression. Use dummy coding for categorical predictors."


def simulate_data(
    dv_type,
    x1_type,
    x2_type=None,
    n_per_cell=30,
    levels_x1=2,
    levels_x2=2,
    slope1=0.6,
    slope2=0.4,
    interaction=0.0,
    noise_sd=1.0,
    seed=123,
):
    rng = np.random.default_rng(seed)

    if x2_type is None:
        if x1_type == "Categorical":
            groups = [f"g{i+1}" for i in range(levels_x1)]
            df = []
            base = 0.0
            step = slope1  # effect magnitude between adjacent group means
            for i, g in enumerate(groups):
                mu = base + i * step
                y = rng.normal(mu, noise_sd, size=n_per_cell)
                df.append(pd.DataFrame({"x1": g, "y": y}))
            data = pd.concat(df, ignore_index=True)
            if dv_type == "Binary":
                # Convert continuous latent to binary via logistic
                p = 1 / (1 + np.exp(-(data["y"])) )
                data["y"] = rng.binomial(1, p)
            return data
        else:
            x = rng.normal(0, 1, size=n_per_cell)
            y_latent = 0.0 + slope1 * x + rng.normal(0, noise_sd, size=n_per_cell)
            if dv_type == "Binary":
                p = 1 / (1 + np.exp(-y_latent))
                y = rng.binomial(1, p)
            else:
                y = y_latent
            return pd.DataFrame({"x1": x, "y": y})
    else:
        # Two predictors
        if x1_type == "Categorical" and x2_type == "Categorical":
            A = [f"A{i+1}" for i in range(levels_x1)]
            B = [f"B{j+1}" for j in range(levels_x2)]
            rows = []
            for i, a in enumerate(A):
                for j, b in enumerate(B):
                    # cell mean = main effects + interaction term
                    mu = i * slope1 + j * slope2 + (i * j) * interaction
                    y = rng.normal(mu, noise_sd, size=n_per_cell)
                    for val in y:
                        rows.append({"x1": a, "x2": b, "y": val})
            data = pd.DataFrame(rows)
            return data
        elif x1_type == "Continuous" and x2_type == "Continuous":
            n = n_per_cell
            x1 = rng.normal(0, 1, size=n)
            x2 = rng.normal(0, 1, size=n)
            y = 0.0 + slope1 * x1 + slope2 * x2 + interaction * (x1 * x2) + rng.normal(0, noise_sd, size=n)
            return pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        else:
            # Mixed case: one categorical, one continuous
            # Ensure x1 is categorical for plotting consistency
            if x1_type == "Continuous":
                # swap to keep x1 categorical
                x1_type, x2_type = x2_type, x1_type
                swapped = True
            else:
                swapped = False
            groups = [f"g{i+1}" for i in range(levels_x1)]
            rows = []
            for i, g in enumerate(groups):
                x2 = rng.normal(0, 1, size=n_per_cell)
                mu = i * slope1 + slope2 * x2 + interaction * (i * x2)
                y = mu + rng.normal(0, noise_sd, size=n_per_cell)
                for xv, yv in zip(x2, y):
                    rows.append({"x1": g, "x2": xv, "y": yv})
            data = pd.DataFrame(rows)
            if swapped:
                data = data.rename(columns={"x1": "x2", "x2": "x1"})
            return data


def run_analysis(df, dv_type, x1_type, x2_type=None):
    results = {}
    if dv_type == "Binary":
        # Logistic regression for any predictor types
        if x2_type is None:
            if x1_type == "Categorical":
                model = smf.logit("y ~ C(x1)", data=df).fit(disp=False)
            else:
                model = smf.logit("y ~ x1", data=df).fit(disp=False)
        else:
            if x1_type == "Categorical" and x2_type == "Categorical":
                model = smf.logit("y ~ C(x1) * C(x2)", data=df).fit(disp=False)
            elif x1_type == "Continuous" and x2_type == "Continuous":
                model = smf.logit("y ~ x1 * x2", data=df).fit(disp=False)
            else:
                model = smf.logit("y ~ C(x1) + x2 + C(x1):x2", data=df).fit(disp=False)
        results["model_summary"] = model.summary2().as_text()
        return results

    # Continuous DV
    if x2_type is None:
        if x1_type == "Categorical":
            model = smf.ols("y ~ C(x1)", data=df).fit()
            aov = anova_lm(model, typ=2)
            results["anova"] = aov
        else:
            model = smf.ols("y ~ x1", data=df).fit()
            results["regression"] = model.summary2().tables[1]
    else:
        if x1_type == "Categorical" and x2_type == "Categorical":
            model = smf.ols("y ~ C(x1) * C(x2)", data=df).fit()
            aov = anova_lm(model, typ=2)
            results["anova"] = aov
        elif x1_type == "Continuous" and x2_type == "Continuous":
            model = smf.ols("y ~ x1 * x2", data=df).fit()
            results["regression"] = model.summary2().tables[1]
        else:
            model = smf.ols("y ~ C(x1) + x2 + C(x1):x2", data=df).fit()
            aov = anova_lm(model, typ=2)
            results["anova"] = aov
            results["regression_note"] = (
                "This ANCOVA equals a regression with dummy coding and interaction."
            )
    return results


def make_plot(df, dv_type, x1_type, x2_type=None):
    if dv_type == "Binary":
        if x2_type is None:
            if x1_type == "Categorical":
                fig = px.bar(df, x="x1", y="y", barmode="group")
            else:
                fig = px.scatter(df, x="x1", y="y")
        else:
            if x1_type == "Categorical" and x2_type == "Categorical":
                fig = px.bar(df, x="x1", y="y", color="x2", barmode="group")
            elif x1_type == "Continuous" and x2_type == "Continuous":
                fig = px.scatter(df, x="x1", y="y", color="x2")
            else:
                fig = px.scatter(df, x="x2", y="y", color="x1")
        return fig

    # Continuous DV
    if x2_type is None:
        if x1_type == "Categorical":
            agg = df.groupby("x1")["y"].agg(["mean", "std", "count"]).reset_index()
            agg["se"] = agg["std"] / np.sqrt(agg["count"])
            fig = px.bar(agg, x="x1", y="mean", error_y="se")
        else:
            fig = px.scatter(df, x="x1", y="y", trendline="ols")
    else:
        if x1_type == "Categorical" and x2_type == "Categorical":
            agg = df.groupby(["x1", "x2"]) ["y"].agg(["mean", "std", "count"]).reset_index()
            agg["se"] = agg["std"] / np.sqrt(agg["count"])
            fig = px.bar(agg, x="x1", y="mean", color="x2", barmode="group", error_y="se")
        elif x1_type == "Continuous" and x2_type == "Continuous":
            fig = px.scatter(df, x="x1", y="y", color="x2", trendline="ols")
        else:
            fig = px.scatter(df, x="x2", y="y", color="x1", trendline="ols")
    return fig

# -----------------------------
# UI
# -----------------------------

with st.sidebar:
    st.header("Design setup")
    dv_type = st.selectbox("DV type", ["Continuous", "Binary"], index=0)
    two_predictors = st.checkbox("Two predictors", value=False)

    if not two_predictors:
        x1_type = st.selectbox("Predictor type", ["Categorical", "Continuous"], index=0)
        x2_type = None
    else:
        x1_type = st.selectbox("Predictor 1", ["Categorical", "Continuous"], index=0)
        x2_type = st.selectbox("Predictor 2", ["Categorical", "Continuous"], index=0)

    st.subheader("Data settings")
    n_per_cell = st.slider("Sample size per cell or total", 10, 300, 60, 10)
    noise_sd = st.slider("Noise SD", 0.1, 3.0, 1.0, 0.1)
    slope1 = st.slider("Effect A or slope 1", 0.0, 2.0, 0.6, 0.1)
    slope2 = st.slider("Effect B or slope 2", 0.0, 2.0, 0.4, 0.1)
    interaction = st.slider("Interaction strength", 0.0, 2.0, 0.0, 0.1)
    seed = st.number_input("Random seed", value=123, step=1)

    if x1_type == "Categorical":
        levels_x1 = st.slider("Levels for factor A", 2, 6, 2, 1)
    else:
        levels_x1 = 2

    if two_predictors and x2_type == "Categorical":
        levels_x2 = st.slider("Levels for factor B", 2, 6, 2, 1)
    else:
        levels_x2 = 2

analysis_msg = recommend_analysis(dv_type, x1_type, x2_type)
st.info(f"Recommended analysis: {analysis_msg}")

# Simulate
if not two_predictors:
    df = simulate_data(
        dv_type, x1_type, None, n_per_cell, levels_x1, 2, slope1, slope2, interaction, noise_sd, seed
    )
else:
    df = simulate_data(
        dv_type, x1_type, x2_type, n_per_cell, levels_x1, levels_x2, slope1, slope2, interaction, noise_sd, seed
    )

st.subheader("Plot")
fig = make_plot(df, dv_type, x1_type, x2_type if two_predictors else None)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Analysis output")
res = run_analysis(df, dv_type, x1_type, x2_type if two_predictors else None)

if "anova" in res:
    st.dataframe(res["anova"].round(4))
if "regression" in res:
    st.dataframe(res["regression"].round(4))
if "model_summary" in res:
    st.text(res["model_summary"])  # logistic
if "regression_note" in res:
    st.caption(res["regression_note"]) 

st.download_button(
    label="Download simulated data (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="simulated_data.csv",
    mime="text/csv",
)

st.markdown(
    """
**How to read this**
- Bar charts with error bars match ANOVA designs with categorical predictors.
- Scatter plots with a trend line match regression designs with continuous predictors.
- Mixed designs appear as grouped scatter plus regression line by group. That is ANCOVA.
- Binary outcomes use logistic models. The plots show 0 or 1 outcomes.
"""
)
