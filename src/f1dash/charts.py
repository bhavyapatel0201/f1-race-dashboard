"""Plotly figure builders. Pure functions: DataFrames in, figure out."""
import plotly.express as px
import plotly.graph_objects as go

TEAM_COLOURS = {
    "red bull": "#3671C6", "ferrari": "#E8002D", "mercedes": "#27F4D2",
    "mclaren": "#FF8000", "aston martin": "#229971", "alpine": "#FF87BC",
    "williams": "#64C4FF", "rb": "#6692FF", "haas": "#B6BABD", "sauber": "#52E252",
}
FALLBACK = "#999999"


def team_colour(team: str) -> str:
    t = team.lower()
    for key, colour in TEAM_COLOURS.items():
        if key in t:
            return colour
    return FALLBACK


def driver_style(results):
    """Map driver code -> (colour, dash). Second driver in a team gets a dashed line."""
    style, seen = {}, {}
    for r in results.sort_values("Finish").itertuples():
        n = seen.get(r.Team, 0)
        seen[r.Team] = n + 1
        style[r.Driver] = (team_colour(r.Team), "solid" if n == 0 else "dash")
    return style


def race_trace(all_laps, results) -> go.Figure:
    style = driver_style(results)
    fig = go.Figure()
    for drv, g in all_laps.sort_values(["Driver", "LapNumber"]).groupby("Driver"):
        colour, dash = style.get(drv, (FALLBACK, "solid"))
        fig.add_trace(go.Scatter(
            x=g["LapNumber"].tolist(), y=g["Position"].astype(int).tolist(),
            mode="lines+markers", name=drv,
            line=dict(width=2, color=colour, dash=dash),
            marker=dict(size=4, color=colour),
            hovertemplate=f"{drv}<br>Lap %{{x}}<br>P%{{y}}<extra></extra>"))
    n = max(20, int(all_laps["Position"].max()))
    fig.update_layout(title="Position by lap (dashed = team's second driver)",
                      xaxis_title="Lap", yaxis_title="Position", height=650)
    fig.update_yaxes(range=[n + 0.5, 0.5], dtick=1)
    fig.update_xaxes(range=[1, int(all_laps["LapNumber"].max())])
    return fig


def strategy(strat, results) -> go.Figure:
    order = results.sort_values("Finish")["Driver"].tolist()
    order = [d for d in order if d in set(strat["Driver"])]
    df = strat.copy()
    df["Stint"] = df["Stint"].astype(str)
    fig = px.bar(df, base="Start", x="Length", y="Driver", color="Stint",
                 orientation="h", title="Stints by lap, drivers in finishing order",
                 category_orders={"Stint": ["1", "2", "3", "4", "5"]},
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 hover_data={"Start": True, "End": True, "Length": True})
    fig.update_yaxes(tickmode="array", tickvals=order, ticktext=order,
                     categoryorder="array", categoryarray=order, autorange="reversed")
    fig.update_layout(xaxis_title="Lap", yaxis_title="", height=680, legend_title="Stint")
    return fig
