"""
Streamlit-based lightweight account tiering dashboard
====================================================

This script provides an interactive application to help you rank and tier
customer accounts based on a few simple characteristics.  It is designed
to work with the ``account_tiering_template.xlsx`` file created alongside
this script, but it can read any CSV or Excel file that contains the
following columns:

``Account``, ``AE``, ``DM``, ``Current/Past Client``, ``Engagement``,
``MSA``, ``Publicly Traded``, ``# Employees``, ``Annual Revenue``, and
``Total Funding``.

How it works
------------

1. Start the app with ``streamlit run account_tiering_app.py``.
2. Upload your account data (an Excel file or CSV) via the sidebar.
3. Adjust the weighting sliders to indicate how important each factor is
   to your overall score.  The weights are expressed as percentages and
   must add up to 1.  There's a handy progress bar showing the total.
4. Choose thresholds for the different tiers.  Scores above the first
   threshold are classified as Tier 1, those between the two thresholds
   as Tier 2, and the rest as Tier 3.
5. Explore the resulting table and charts: a sortable score table, a
   distribution of tiers, and a scatter plot showing how revenue and
   headcount relate to one another.

This app does not send your data anywhere – everything runs locally on
your machine.  You can export the scored results as a CSV if you like.

"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt


def normalize(series: pd.Series) -> pd.Series:
    """Normalize a numeric series to the range 0–1.

    Missing or non-positive values are treated as zero.  If all values
    are zero or missing, the returned series will also be all zeros.
    """
    s = series.fillna(0).astype(float)
    max_val = s.max()
    return s / max_val if max_val > 0 else s


@st.cache_data
def load_data(file: object) -> pd.DataFrame:
    """Load an uploaded file into a DataFrame.

    Supports CSV, XLSX, and XLS files.  Raises a ValueError if the
    required columns are missing.
    """
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    expected_cols = [
        "Account",
        "AE",
        "DM",
        "Current/Past Client",
        "Engagement",
        "MSA",
        "Publicly Traded",
        "# Employees",
        "Annual Revenue",
        "Total Funding",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following required columns are missing from your file: {', '.join(missing)}"
        )
    return df


def compute_scores(
    df: pd.DataFrame,
    weights: dict[str, float],
    tier1_threshold: float,
    tier2_threshold: float,
) -> pd.DataFrame:
    """Compute normalized metrics, overall scores, and tiers.

    The weights dictionary should have the keys ``employees``, ``revenue``,
    ``funding``, ``current_past``, ``engagement``, ``msa``, and
    ``publicly_traded``.  Thresholds define boundaries between tiers.
    """
    out = df.copy()

    # Normalize numeric metrics
    out["norm_employees"] = normalize(out["# Employees"])
    out["norm_revenue"] = normalize(out["Annual Revenue"])
    out["norm_funding"] = normalize(out["Total Funding"])

    # Binary conversions (Yes→1, everything else→0)
    out["bin_current"] = out["Current/Past Client"].str.strip().str.lower().eq("yes").astype(float)
    out["bin_engagement"] = out["Engagement"].str.strip().str.lower().eq("yes").astype(float)
    out["bin_msa"] = out["MSA"].str.strip().str.lower().eq("yes").astype(float)
    out["bin_public"] = out["Publicly Traded"].str.strip().str.lower().eq("yes").astype(float)

    # Compute score
    out["Score"] = (
        weights["employees"] * out["norm_employees"]
        + weights["revenue"] * out["norm_revenue"]
        + weights["funding"] * out["norm_funding"]
        + weights["current_past"] * out["bin_current"]
        + weights["engagement"] * out["bin_engagement"]
        + weights["msa"] * out["bin_msa"]
        + weights["publicly_traded"] * out["bin_public"]
    )

    # Assign tier
    conditions = [out["Score"] >= tier1_threshold, out["Score"] >= tier2_threshold]
    choices = ["Tier 1", "Tier 2"]
    out["Tier"] = np.select(conditions, choices, default="Tier 3")

    return out


def main() -> None:
    st.set_page_config(page_title="Account Tiering Dashboard", layout="wide")
    st.title("Account Tiering Dashboard")
    st.markdown(
        """
        This dashboard helps you classify and rank your accounts based on
        several factors.  Start by uploading your data file in the sidebar,
        then adjust the weighting and tier thresholds to suit your needs.
        """
    )

    with st.sidebar:
        st.header("Data & Settings")
        data_file = st.file_uploader(
            "Upload data (CSV or Excel)", type=["csv", "xlsx", "xls"], key="file"
        )
        # Allow loading data directly from a Google Sheet
        # Users can paste the CSV export link from Google Sheets here.  For example:
        # https://docs.google.com/spreadsheets/d/<sheet-id>/export?format=csv
        st.markdown("### Or load from Google Sheets")
        google_sheet_url = st.text_input(
            "Google Sheets CSV URL",
            help="Paste the export link obtained via File → Share → Publish to web → Comma‑separated values (csv)."
        )

        st.markdown("### Weights (total must equal 1)")
        employees_weight = st.number_input("Employees", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        revenue_weight = st.number_input("Annual Revenue", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        funding_weight = st.number_input("Total Funding", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        current_past_weight = st.number_input("Current/Past Client", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
        engagement_weight = st.number_input("Engagement", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
        msa_weight = st.number_input("MSA", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        public_weight = st.number_input("Publicly Traded", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        total_weight = employees_weight + revenue_weight + funding_weight + current_past_weight + engagement_weight + msa_weight + public_weight
        st.progress(min(total_weight, 1.0))
        if abs(total_weight - 1.0) > 1e-6:
            st.error(f"Total weight is {total_weight:.2f}, but it must equal 1.")

        st.markdown("### Tier thresholds")
        tier1 = st.slider("Tier 1 threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
        tier2 = st.slider("Tier 2 threshold", min_value=0.0, max_value=tier1, value=0.5, step=0.01)

    # Load the DataFrame either from Google Sheets or from the uploaded file.
    df = None
    if google_sheet_url:
        try:
            # Read the CSV directly from the provided URL
            df = pd.read_csv(google_sheet_url)
            # Ensure required columns exist
            expected_cols = [
                "Account",
                "AE",
                "DM",
                "Current/Past Client",
                "Engagement",
                "MSA",
                "Publicly Traded",
                "# Employees",
                "Annual Revenue",
                "Total Funding",
            ]
            missing_cols = [c for c in expected_cols if c not in df.columns]
            if missing_cols:
                st.error(
                    f"The following required columns are missing from your Google Sheet: {', '.join(missing_cols)}"
                )
                df = None
        except Exception as e:
            st.error(f"Failed to load data from Google Sheets: {e}")
            df = None
    elif data_file is not None:
        try:
            df = load_data(data_file)
        except Exception as e:
            st.error(str(e))
            df = None

    # Only proceed if we have data and the total weight is valid
    if df is not None:
        if abs(total_weight - 1.0) > 1e-6:
            st.error(f"Total weight is {total_weight:.2f}, but it must equal 1.")
        else:
            weights = {
                "employees": employees_weight,
                "revenue": revenue_weight,
                "funding": funding_weight,
                "current_past": current_past_weight,
                "engagement": engagement_weight,
                "msa": msa_weight,
                "publicly_traded": public_weight,
            }

            scored = compute_scores(df, weights, tier1, tier2)

            # Display results
            st.subheader("Scored Accounts")
            st.dataframe(
                scored[
                    [
                        "Account",
                        "AE",
                        "DM",
                        "Score",
                        "Tier",
                        "# Employees",
                        "Annual Revenue",
                        "Total Funding",
                    ]
                ]
                .sort_values("Score", ascending=False)
                .reset_index(drop=True)
            )

            st.subheader("Tier Distribution")
            tier_counts = (
                scored["Tier"]
                .value_counts()
                .rename_axis("Tier")
                .reset_index(name="Count")
            )
            bar = (
                alt.Chart(tier_counts)
                .mark_bar()
                .encode(x="Tier", y="Count", color="Tier")
                .properties(width=400)
            )
            st.altair_chart(bar, use_container_width=True)

            st.subheader("Revenue vs Employees")
            scatter = (
                alt.Chart(scored)
                .mark_circle(size=60)
                .encode(
                    x=alt.X(
                        "# Employees", scale=alt.Scale(type="log"), title="# Employees (log scale)"
                    ),
                    y=alt.Y(
                        "Annual Revenue", scale=alt.Scale(type="log"), title="Annual Revenue (log scale)"
                    ),
                    color="Tier",
                    tooltip=["Account", "AE", "DM", "Score", "Tier"]
                )
                .interactive()
            )
            st.altair_chart(scatter, use_container_width=True)

            # Option to download results
            csv_bytes = scored.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download scored data as CSV",
                data=csv_bytes,
                file_name="scored_accounts.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
