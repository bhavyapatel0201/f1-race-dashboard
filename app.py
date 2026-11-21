"""Run: streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from f1dash.analysis import (add_stints, clean_laps, lap_gap, stint_summary,
                             strategy_table, track_evolution)
from f1dash.charts import race_trace, strategy
from f1dash.data import get_schedule, load_race

st.set_page_config(page_title="F1 Race Dashboard", layout="wide")
st.title("F1 Race Data Dashboard")
st.caption("Data: Jolpica-F1 (free, no login). Stints are inferred from pit stops.")


@st.cache_data(show_spinner="Loading schedule...")
def schedule(year):
    return get_schedule(year)


@st.cache_data(show_spinner="Downloading race data (first load takes a few seconds)...")
def race(year, rnd):
    return load_race(year, rnd)


with st.sidebar:
    year = int(st.number_input("Season", 1996, 2026, 2024))
    sched = schedule(year)
    label = st.selectbox("Race", [f"{r.Round}. {r.Name}" for r in sched.itertuples()])
    rnd = int(label.split(".")[0])

data = race(year, rnd)
results, pits = data["results"], data["pits"]
all_laps = add_stints(data["laps"], pits)
laps = clean_laps(all_laps)
drivers = sorted(all_laps["Driver"].unique())

st.subheader(label)
tabs = st.tabs(["Results", "Race trace", "Driver gap", "Strategy", "Stint pace", "Track evolution"])

with tabs[0]:
    st.dataframe(results[["Finish", "Grid", "Driver", "Team", "Laps", "Status"]]
                 .sort_values("Finish"), hide_index=True, use_container_width=True)

with tabs[1]:
    st.plotly_chart(race_trace(all_laps, results), use_container_width=True)

with tabs[2]:
    c1, c2 = st.columns(2)
    a = c1.selectbox("Driver A", drivers, index=0)
    b = c2.selectbox("Driver B", drivers, index=min(1, len(drivers) - 1))
    st.plotly_chart(
        px.line(lap_gap(laps, a, b), x="LapNumber", y="CumulativeGap",
                title=f"Cumulative gap {a} vs {b} (s; negative = {a} faster)"),
        use_container_width=True)

with tabs[3]:
    strat = strategy_table(all_laps[~all_laps["LapTimeSeconds"].isna()])
    st.plotly_chart(strategy(strat, results), use_container_width=True)
    if len(pits):
        st.caption(f"Median pit-lane time: {pits['PitSeconds'].median():.1f} s")

with tabs[4]:
    summary = stint_summary(laps)
    if summary.empty:
        st.info("Not enough clean laps per stint for this race.")
    else:
        plot = summary.assign(Stint=summary["Stint"].astype(str))
        fig = px.bar(plot, x="Driver", y="DegPerLap", color="Stint", barmode="group",
                     category_orders={"Stint": ["1", "2", "3", "4", "5"]},
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     hover_data=["Laps", "MedianPace"],
                     title="Seconds lost per lap of tyre age (stints of 8+ clean laps; includes fuel burn-off)")
        fig.update_layout(yaxis_title="Seconds per lap of tyre age", legend_title="Stint")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(summary, hide_index=True, use_container_width=True)

with tabs[5]:
    evo = track_evolution(laps)
    st.plotly_chart(px.line(evo, x="LapNumber", y=["Median", "Smoothed"],
                            title="Field median lap time"), use_container_width=True)
