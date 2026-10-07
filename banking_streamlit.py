import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. يجب أن يكون هذا أول أمر لـ Streamlit دائماً
st.set_page_config(
    page_title="Customer Churn Analytics",
    layout="wide"
)

# غيّر الـ path دا لو ملفك في مكان تاني
DATA_PATH = r"c:\Users\ELZAHBIA\AppData\Local\Temp\Rar$DRa13972.11815\Churn_Modelling.csv"

# ====================== FINAL MESSAGE (اكتب رسالتك هنا لو عايزها تثبت) ======================
# اللي بتكتبه جوه التطبيق بيضيع لو عملت refresh للصفحة.
# لو عايز الرسالة تثبت، اكتبها هنا بين الـ """ """.
FINAL_MESSAGE = """
Overall, 20.4% of our customers left the bank (2,037 of 10,000). The risk is concentrated in a few groups.

1. Age 49-60 is the biggest risk: 55.1% churn, 2.7x the bank average.
   Action: proactive retention calls to this group, starting in Germany.

2. Germany: 32.4% churn vs about 16% in France and Spain, for both men and women.
   Action: survey customers who left in Germany to find out why, before changing anything.

3. Women churn more than men (25.1% vs 16.5%) in all three countries, and age does not explain it.
   Action: ask leavers why, to find what we can improve.

4. Customers with 1 product are half of the bank and 69.2% of all leavers (27.7% churn). Customers with 2 products churn only 7.6%.
   Action: offer a 2nd product to a sample of 1-product customers, and compare their churn with the rest (A/B test).

5. Inactive customers churn about 1.9x more than active ones (26.85% vs 14.27%).
   Action: use falling activity as an early-warning signal for retention.

6. Customers with 100k+ balance churn above average (25.8% and 23.0%) and hold the most money.
   Action: give them priority retention attention.

Limits: the data shows who leaves, not why. Activity and product count are associations, not proven causes. Customers with 3-4 products are too few to draw firm conclusions.
"""

# ====================== CONCLUSIONS (الاستنتاجات اللي وصلنا لها) ======================
CONCLUSIONS = {
    "country": """
- **Germany's churn rate is 32.4%**, about double France (16.2%) and Spain (16.7%).
- Germany is high for **both genders**: German men churn at 27.8%, more than French women (20.3%).
- The chart shows counts. Judge countries by **rate**, not by count.
- The note on the Germany bar shows how many German leavers are male vs female.
""",
    "gender": """
- **Women churn at 25.1%**, men at 16.5%.
- The gap exists in **all three countries**: France 20.3% vs 12.7%, Spain 21.2% vs 13.1%, Germany 37.6% vs 27.8%.
- Age does **not** explain it: in Germany the average age is 40.2 for women vs 39.4 for men, and women churn more in every age band.
- So country and gender are two **independent** risk factors.
""",
    "gender_germany": """
- This chart shows the **number** of German women who left, by age (ages with more than 10 leavers only).
- About **40%** of German women who left are aged 39-48.
- Counts show **where the leavers are**, not the risk at each age. Fewer older customers exist, so counts fall at older ages.
- To judge risk, compare **rates** inside each age band. Women churn more than men in every age band in Germany.
""",
    "age": """
- Customers aged **49-60** churn at **55.1%** (1,078 customers), versus 20.4% for the whole bank.
- The line chart shows **counts**. Counts fall after the mid-40s because there are fewer older customers, not because churn falls.
- The churn **rate** stays above 40% at every age from 49 to 60.
""",
    "age_vs_overall": """
- A customer aged 49-60 is about **2.7x more likely to leave** than the average customer (55.1% vs 20.4%).
- This age group is **29.2% of all churned customers**. That is their share of leavers, not their churn rate.
""",
    "balance": """
- Churn rate by balance group (n = customers in the group):
  - **0 balance:** 13.8% (n=3,617), the lowest. This group is 36% of the bank.
  - **100k-150k:** 25.8% (n=3,830), the highest among large groups.
  - **150k+:** 23.0% (n=968). **50k-100k:** 19.9% (n=1,509).
- Customers with 100k+ balance churn above the bank average (20.4%), and they hold the most money, so retention is worth the effort.
- The gap is **moderate**, much smaller than the age effect.
- **1-50k** shows 34.7% but has only **75 customers**, so we do not draw conclusions from it.
""",
    "products": """
- **1 product:** 5,084 customers (50.8% of the bank), churn 27.7%. They are **69.2% of all churned customers**.
- **2 products:** churn only 7.6%. This is the most stable group.
- **3 products:** 82.7% (n=266). **4 products:** 100% (n=60). Small groups, so read this as a direction only.
- Idea: encourage 1-product customers to take a 2nd product (cross-selling). This is a **hypothesis**, not proven. Test it with an A/B test.
""",
    "active_member": """
- **Inactive:** 26.85% churn (4,849 customers). **Active:** 14.27% (5,151 customers).
- Inactive customers leave about **1.9x** more often.
- This is an **association, not a proven cause**. Inactivity may be an early-warning signal.
- Whether re-activating customers keeps them can only be confirmed with an A/B test.
""",
    "classification_report": """
- Accuracy is about 0.87, but the baseline "everyone stays" already gives about 0.80, so accuracy alone is misleading.
- **Recall for churners is about 0.4**: the model catches about 4 out of 10 customers who leave.
- Precision for churners is about 0.8: when it flags a customer, it is usually right.
- Adding Age, Geography and Gender improved churn recall from about 0.15 to about 0.4.
""",
    "slope_graph": """
- The model predicts **fewer churners than actually exist**: it misses roughly 370-385 of the 643 real churners in the test set.
- Business meaning: the model is a good shortlist for retention calls, but it does not find every leaver.
""",
}

# ====================== CSS STYLING ======================
st.markdown(
    """
<style>
    /* تغيير خلفية التطبيق بالكامل */
    .stApp {
        background-color: #0e1117;
    }

    /* إجبار جميع النصوص على اللون الأبيض */
    .stApp, .stMarkdown, .stMetric, h1, h2, h3, h4, p, label, .css-1d391kg, .st-emotion-cache {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #cbd5e1 !important;
    }

    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 15px;
        margin-bottom: 30px;
    }
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 12px;
        padding: 20px 10px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .kpi-card:hover {
        transform: translateY(-8px);
        border-color: #60a5fa;
        box-shadow: 0 12px 25px rgba(59, 130, 246, 0.4);
        background: rgba(59, 130, 246, 0.15);
    }
    .kpi-icon {
        font-size: 32px;
        margin-bottom: 10px;
        display: block;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin: 5px 0;
    }
    .kpi-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .gradient-title {
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(90deg, #45e7ff, #7f8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent !important;
        margin: 10px 0;
        display: inline-block;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ====================== HEADER ======================
st.markdown('<h1 class="gradient-title">📊 Customer Churn Analytics</h1>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#bae6fd; margin:0;'>Customer churn analysis dashboard</p>",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Helper: renders a text_area + a live markdown preview under any chart
# ----------------------------------------------------------------------
def conclusion_box(key: str, default: str = ""):
    st.markdown("**📝 Conclusion / Recommendation**")
    with st.expander("✏️ Edit conclusion"):
        text = st.text_area(
            label="Conclusion",
            key=f"conclusion_{key}",
            value=default.strip(),
            height=180,
            label_visibility="collapsed",
        )
    if text.strip():
        with st.container(border=True):
            st.markdown(text)


# ----------------------------------------------------------------------
# Helper: dark theme for plotly figures
# ----------------------------------------------------------------------
def apply_modern_layout(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#ffffff"),
        title=dict(font=dict(size=16, family="Arial, sans-serif", color="#ffffff"), x=0, y=0.95),
        margin=dict(l=40, r=40, t=60, b=50),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            title_text="", font=dict(color="#ffffff", size=12),
        ),
        hoverlabel=dict(
            bgcolor="#1e293b", font_size=12, font_family="Inter, sans-serif",
            bordercolor="rgba(255,255,255,0.1)", font_color="#ffffff",
        ),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color="#ffffff"), title_font=dict(color="#ffffff"), linecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#ffffff"), title_font=dict(color="#ffffff"), zeroline=False)
    return fig


# ----------------------------------------------------------------------
# Helper: dark theme for matplotlib figures
# ----------------------------------------------------------------------
def style_mpl_dark(fig, ax):
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    return fig, ax


COLOR_MAP = {"Stay": "#94a3b8", "Exited": "#e74c3c"}


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    if "RowNumber" in df.columns:
        df = df.drop(columns=["RowNumber"])
    return df


try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"الملف `{DATA_PATH}` مش موجود. حطه جنب السكريبت أو عدّل DATA_PATH.")
    st.stop()

# ========================================================================
# Sidebar navigation
# ========================================================================
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "اذهب إلى",
    ["Country", "Gender", "Age", "Balance", "Products", "Active Member", "Model"],
)

# ========================================================================
# KPIs (always visible on top, modern card layout)
# ========================================================================
total_customers = len(df)
total_exited = int(df["Exited"].sum())
exited_pct = (total_exited / total_customers) * 100 if total_customers else 0
active_pct = (df["IsActiveMember"] == 1).mean() * 100
inactive_pct = 100 - active_pct

kpi_html = f"""
<div class="kpi-container">
    <div class="kpi-card">
        <span class="kpi-icon">👥</span>
        <div class="kpi-value">{total_customers:,}</div>
        <div class="kpi-label">Total Customers</div>
    </div>
    <div class="kpi-card" style="border-color: rgba(239, 68, 68, 0.5);">
        <span class="kpi-icon" style="color: #ef4444;">🚪</span>
        <div class="kpi-value" style="color: #ef4444;">{exited_pct:.1f}%</div>
        <div class="kpi-label">Total Exited</div>
    </div>
    <div class="kpi-card" style="border-color: rgba(59, 130, 246, 0.5);">
        <span class="kpi-icon" style="color: #3b82f6;">✅</span>
        <div class="kpi-value" style="color: #3b82f6;">{active_pct:.1f}%</div>
        <div class="kpi-label">Active</div>
    </div>
    <div class="kpi-card">
        <span class="kpi-icon">💤</span>
        <div class="kpi-value">{inactive_pct:.1f}%</div>
        <div class="kpi-label">Inactive</div>
    </div>
</div>
"""

# expanded=True: مفتوحة في الأول. اضغط على العنوان علشان تخفيها وتفتحها.
with st.expander("💬 Final Message", expanded=True):
    final_message = st.text_area(
        label="Final Message",
        key="final_message",
        value=FINAL_MESSAGE,
        placeholder="اكتب الـ Final Message هنا...",
        height=160,
        label_visibility="collapsed",
    )
    if final_message.strip():
        with st.container(border=True):
            st.markdown(final_message)

st.markdown("### KPI")
st.markdown(kpi_html, unsafe_allow_html=True)

# ========================================================================
# Country
# ========================================================================
if page == "Country":
    st.header("Churn Count per Country")

    churn_data = df.groupby(["Geography", "Exited"]).size().reset_index(name="Count")
    churn_data["Status"] = churn_data["Exited"].map({0: "Stay", 1: "Exited"})

    country_order = ["France", "Germany", "Spain"]
    churn_data["Geography"] = pd.Categorical(churn_data["Geography"], categories=country_order, ordered=True)
    churn_data = churn_data.sort_values("Geography")

    fig1 = px.bar(
        churn_data,
        x="Count",
        y="Geography",
        color="Geography",
        orientation="h",
        barmode="group",
        text="Count",
        title="Churn Count per Country",
        category_orders={"Geography": country_order},
        color_discrete_map={"Germany": "#B71C1C", "France": "#E53935", "Spain": "#EF9A9A"},
    )
    fig1 = apply_modern_layout(fig1)
    fig1.update_traces(textposition="outside")
    fig1.update_layout(
        xaxis=dict(visible=False),
        showlegend=False,
        title=dict(text="Churn Count per Country", x=0.01, y=0.98, xanchor="left", yanchor="top"),
    )
    fig1.add_shape(
        type="rect",
        xref="paper", yref="y",
        x0=0, x1=0.51, y0=0.5, y1=1.5,
        line=dict(color="#e74c3c", width=3, dash="dash"),
        fillcolor="rgba(0,0,0,0)",
    )
    fig1.update_yaxes(title=None, showticklabels=True)

    # عدد الرجالة والستات اللي مشيوا من ألمانيا
    germany_exited = df[(df["Geography"] == "Germany") & (df["Exited"] == 1)]
    gender_counts = germany_exited["Gender"].value_counts()
    gender_text = f"Male: {gender_counts.get('Male', 0)} | Female: {gender_counts.get('Female', 0)}"
    fig1.add_annotation(
        x=gender_counts.sum() / 2,
        y="Germany",
        text=gender_text,
        showarrow=False,
        font=dict(size=13, color="white"),
    )
    st.plotly_chart(fig1, use_container_width=True)
    conclusion_box("country", CONCLUSIONS["country"])

# ========================================================================
# Gender
# ========================================================================
elif page == "Gender":
    st.header("Churn Rate Distribution by Gender")

    churn_per_sex = (
        df.groupby("Gender")["Exited"].value_counts(normalize=True).mul(100).reset_index(name="Percentage")
    )
    churn_per_sex["Status"] = churn_per_sex["Exited"].map({0: "Stay", 1: "Exited"})
    churn_per_sex["pct_label"] = churn_per_sex["Percentage"].apply(lambda x: f"{x:.1f}%")

    fig2 = px.bar(
        churn_per_sex,
        x="Gender",
        y="Percentage",
        color="Status",
        text="pct_label",
        title="Churn Rate Distribution by Gender",
        color_discrete_map=COLOR_MAP,
    )
    fig2 = apply_modern_layout(fig2)
    fig2.update_traces(textposition="inside")
    fig2.update_xaxes(title="Gender")
    fig2.update_yaxes(visible=False)
    fig2.update_layout(
        barmode="stack",
        legend_title_text="Status",
        legend=dict(yanchor="top", y=1.1, xanchor="left", x=0.01, orientation="h"),
    )
    st.plotly_chart(fig2, use_container_width=True)
    conclusion_box("gender", CONCLUSIONS["gender"])

    st.subheader("Germany: female churners by age")

    female_age = (
        df[(df["Geography"] == "Germany") & (df["Exited"] == 1) & (df["Gender"] == "Female")]
        .groupby("Age")
        .size()
        .reset_index(name="Count")
    )
    total_female_germany = df[(df["Geography"] == "Germany") & (df["Gender"] == "Female")].shape[0]
    female_age["Percentage"] = (female_age["Count"] / total_female_germany) * 100

    focus = female_age[female_age["Count"] > 10].reset_index(drop=True)
    peak = focus[focus["Age"].between(39, 48)]

    fig_fa = px.scatter(
        focus,
        x="Age",
        y="Count",
        title="Germany Female Churn by Age",
        labels={"Age": "Age", "Count": "Female Churn Count"},
        text="Count",
    )
    fig_fa.update_traces(textposition="top center", marker=dict(color="#94a3b8", size=9))
    fig_fa.add_scatter(
        x=peak["Age"],
        y=peak["Count"],
        mode="markers+text",
        marker=dict(color="#2ecc71", size=10),
        text=peak["Count"],
        textposition="top center",
        showlegend=False,
    )
    fig_fa = apply_modern_layout(fig_fa)
    fig_fa.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig_fa.add_vrect(
        x0=39, x1=48,
        fillcolor="#e74c3c",
        opacity=0.12,
        line_width=2,
        line_color="#e74c3c",
        line_dash="dash",
        layer="below",
        annotation_text="Peak Churn (39-48)",
        annotation_position="top left",
        annotation_font=dict(size=12, color="#e74c3c"),
    )
    st.plotly_chart(fig_fa, use_container_width=True)
    conclusion_box("gender_germany", CONCLUSIONS["gender_germany"])

# ========================================================================
# Age
# ========================================================================
elif page == "Age":
    st.header("Customer Churn Distribution by Age")

    churn_per_age = df.groupby("Age")["Exited"].value_counts().reset_index(name="Count")
    churn_per_age["Status"] = churn_per_age["Exited"].map({0: "Stay", 1: "Exited"})
    churn_per_age = churn_per_age.sort_values("Age")

    churn_per_age["Total_Age"] = churn_per_age.groupby("Age")["Count"].transform("sum")
    churn_per_age["Percentage"] = (churn_per_age["Count"] / churn_per_age["Total_Age"]) * 100
    fig3 = px.line(
        churn_per_age,
        x="Age",
        y="Count",
        color="Status",
        markers=False,
        custom_data=["Percentage", "Total_Age"],
        title="Customer Churn Distribution by Age",
        color_discrete_map=COLOR_MAP,
    )
    fig3.update_traces(
        hovertemplate="<br>".join(
            [
                "<b>Age:</b> %{x}",
                "<b>Count:</b> %{y}",
                "<b>Percentage:</b> %{customdata[0]:.1f}%",
                "<b>Total in Age Group:</b> %{customdata[1]}",
            ]
        )
    )

    # نسبة الـ churn فوق كل سن في المنطقة 49-60
    peak_data = churn_per_age[churn_per_age["Age"].between(49, 60) & (churn_per_age["Exited"] == 1)]
    fig3.add_scatter(
        x=peak_data["Age"],
        y=peak_data["Count"],
        mode="text",
        text=peak_data["Percentage"].map(lambda x: f"{x:.1f}%"),
        textposition="top center",
        textfont=dict(color="#ffffff", size=11),
        showlegend=False,
        hoverinfo="skip",
    )

    fig3 = apply_modern_layout(fig3)
    fig3.update_layout(
        xaxis_title="Age",
        yaxis_title="Number of Customers",
        hovermode="x unified",
        legend_title_text="Status",
        title=dict(text="Customer Churn Distribution by Age", x=0.01, y=0.98, xanchor="left", yanchor="top"),
        legend=dict(yanchor="top", y=1.2, xanchor="left", x=0.01, orientation="h"),
    )
    fig3.update_xaxes(showgrid=True)
    fig3.add_vrect(
        x0=49, x1=60,
        fillcolor="#e74c3c",
        opacity=0.12,
        line_width=2,
        line_color="#e74c3c",
        line_dash="dash",
        layer="below",
        annotation_text="Peak Churn (49-60)",
        annotation_position="top left",
        annotation_font=dict(size=12, color="#e74c3c"),
    )
    st.plotly_chart(fig3, use_container_width=True)
    conclusion_box("age", CONCLUSIONS["age"])

    df_49_60 = df[df["Age"].between(49, 60)]
    group_rate = df_49_60["Exited"].mean() * 100
    overall_rate = df["Exited"].mean() * 100
    share_of_exited = df_49_60["Exited"].sum() / df["Exited"].sum() * 100

    plot_df = pd.DataFrame(
        {"Group": ["Customers aged 49-60", "All customers"], "Churn Rate": [group_rate, overall_rate]}
    )
    plot_df["label"] = plot_df["Churn Rate"].apply(lambda x: f"{x:.1f}%")

    fig5 = px.bar(
        plot_df,
        x="Group",
        y="Churn Rate",
        text="label",
        color="Group",
        color_discrete_map={"Customers aged 49-60": "#f87171", "All customers": "#64748b"},
        title=f"Customers aged 49-60 are {group_rate / overall_rate:.1f}x more likely to leave than average",
    )
    fig5 = apply_modern_layout(fig5)
    fig5.update_traces(textposition="outside")
    fig5.add_annotation(
        x=0.98,
        y=1.0,
        xref="paper",
        yref="paper",
        text=f"{share_of_exited:.1f}% of all churned customers<br>are in this age group",
        showarrow=False,
        align="left",
        font=dict(size=13, color="#94a3b8"),
    )
    fig5.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title="Churn Rate (%)",
        yaxis_range=[0, 70],
    )
    st.plotly_chart(fig5, use_container_width=True)
    conclusion_box("age_vs_overall", CONCLUSIONS["age_vs_overall"])

# ========================================================================
# Balance
# ========================================================================
elif page == "Balance":
    st.header("Churn Rate by Balance Group")

    df_bal = df.copy()
    df_bal["Balance_Group"] = pd.cut(
        df_bal["Balance"],
        bins=[-1, 0, 50000, 100000, 150000, 250000],
        labels=["0 Balance", "1 - 50k", "50k - 100k", "100k - 150k", "150k+"],
    )

    bal = df_bal.groupby("Balance_Group", observed=False)["Exited"].agg(rate="mean", n="count").reset_index()
    bal["rate"] = bal["rate"] * 100
    overall_bal = df["Exited"].mean() * 100
    bal["label_x"] = bal.apply(lambda r: f"{r['Balance_Group']}<br>(n={int(r['n']):,})", axis=1)

    # نلوّن أعلى شريحة بس بين الشرائح الكبيرة (500 عميل فأكتر)
    big = bal[bal["n"] >= 500]
    top_group = big.loc[big["rate"].idxmax(), "Balance_Group"]
    low_group = big.loc[big["rate"].idxmin(), "Balance_Group"]
    top_rate = big["rate"].max()
    low_rate = big["rate"].min()
    bal_colors = ["#e74c3c" if g == top_group else "#64748b" for g in bal["Balance_Group"]]

    fig6 = go.Figure(
        go.Bar(
            x=bal["label_x"],
            y=bal["rate"],
            marker_color=bal_colors,
            text=bal["rate"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
        )
    )
    fig6 = apply_modern_layout(fig6)
    fig6.add_hline(
        y=overall_bal,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text=f"Bank average: {overall_bal:.1f}%",
        annotation_position="top left",
        annotation_font_color="#ffffff",
    )
    fig6.update_layout(
        title=dict(
            text=(
                f"Customers with {top_group} balance churn at {top_rate:.1f}%, "
                f"almost {top_rate / low_rate:.1f}x the rate of {low_group} customers ({low_rate:.1f}%)"
            ),
            x=0.01,
            xanchor="left",
        ),
        showlegend=False,
        xaxis_title="Balance Range",
        yaxis_title=None,
        yaxis_range=[0, 45],
    )
    fig6.update_yaxes(showgrid=False, visible=False)
    st.plotly_chart(fig6, use_container_width=True)
    conclusion_box("balance", CONCLUSIONS["balance"])

# ========================================================================
# Products
# ========================================================================
elif page == "Products":
    st.header("Churn Rate by Number of Products")

    prod = df.groupby("NumOfProducts")["Exited"].agg(rate="mean", n="count").reset_index()
    prod["rate"] = prod["rate"] * 100
    overall = df["Exited"].mean() * 100
    prod["label_x"] = prod.apply(lambda r: f"{int(r['NumOfProducts'])} products<br>(n={int(r['n']):,})", axis=1)
    prod["NumOfProducts"] = pd.Categorical(prod["NumOfProducts"], categories=[4, 3, 2, 1], ordered=True)
    prod = prod.sort_values("NumOfProducts")

    blue_colors = {1: "#0057B8", 3: "#9DC3E6", 4: "#D9EAF7", 2: "#5B9BD5"}
    n_products = prod["NumOfProducts"].astype(int)
    bar_colors = n_products.map(blue_colors).tolist()
    text_colors = ["#0f172a" if n in (3, 4) else "#ffffff" for n in n_products]

    customers_prod_1 = ((df["Exited"] == 1) & (df["NumOfProducts"] == 1)).sum()
    total = df["Exited"].value_counts().sum()
    total_prod_1 = (df["NumOfProducts"] == 1).sum()
    perc = (customers_prod_1 / (df["Exited"] == 1).sum()) * 100
    final_calc = (customers_prod_1 / total) * 100
    final_prod_1 = (total_prod_1 / total) * 100

    col1, col2 = st.columns([2, 1])

    with col1:
        fig7 = go.Figure(
            go.Bar(
                x=prod["rate"],
                y=prod["label_x"],
                marker_color=bar_colors,
                text=prod["rate"].apply(lambda x: f"{x:.1f}%"),
                textposition="inside",
                textfont=dict(color=text_colors),
                orientation="h",
            )
        )
        fig7 = apply_modern_layout(fig7)
        fig7.add_vline(
            x=overall,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"Bank average: {overall:.1f}%",
            annotation_position="top",
            annotation_font_color="#ffffff",
        )
        fig7.update_layout(
            title=dict(text="Churn Rate by Number of Products", x=0.01, xanchor="left"),
            showlegend=False,
            xaxis_title="Churn Rate (%)",
            yaxis_title=None,
            xaxis_range=[0, 100],
        )
        fig7.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        fig7.update_yaxes(showgrid=False)
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        st.metric("Product-1 customers", f"{final_prod_1:.1f}%", help="من كل عملاء البنك")
        st.metric("Of all churned customers", f"{perc:.1f}%", help="عملاء منتج واحد بس")
        st.metric("Churned & product-1 / whole bank", f"{final_calc:.1f}%")

    conclusion_box("products", CONCLUSIONS["products"])

# ========================================================================
# Active Member
# ========================================================================
elif page == "Active Member":
    st.header("Churn Percentage by Active Member")

    active_pers = pd.crosstab(df["IsActiveMember"], df["Exited"], normalize="index") * 100

    fig8, ax8 = plt.subplots(figsize=(6, 4))
    sns.heatmap(active_pers, annot=True, fmt=".2f", cmap="Blues", ax=ax8, cbar_kws={"label": ""})
    style_mpl_dark(fig8, ax8)
    ax8.set_title("Churn Percentage by Active Member")
    ax8.set_xlabel("Exited")
    ax8.set_ylabel("IsActiveMember")
    st.pyplot(fig8)
    conclusion_box("active_member", CONCLUSIONS["active_member"])

# ========================================================================
# Model
# ========================================================================
elif page == "Model":
    st.header("Modeling — Random Forest Classifier")

    df3 = df.copy()
    new_df = pd.get_dummies(df3.iloc[:, 3:5], dtype=int)
    df3 = pd.concat([df3.iloc[:, :3], new_df, df3.iloc[:, 5:]], axis=1)

    # نفس الـ features اللي في الـ notebook، بس بالاسم بدل الـ index
    drop_cols = ["CustomerId", "Surname", "CreditScore", "EstimatedSalary", "Exited", "Balance_Group"]
    X = df3.drop(columns=drop_cols, errors="ignore").to_numpy()
    y = df3["Exited"].to_numpy()

    min_max = MinMaxScaler()
    X = min_max.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33)

    @st.cache_resource
    def train_model(X_train, y_train):
        model = RandomForestClassifier(n_estimators=400, max_depth=7, min_samples_split=3)
        model.fit(X_train, y_train)
        return model

    RFC = train_model(X_train, y_train)
    y_pred = RFC.predict(X_test)

    aas = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    st.metric("Accuracy", f"{aas:.2f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classification Report")
        cr_df = pd.DataFrame(cr).T
        report_metrics = cr_df.iloc[:-3, :-1]

        fig_cr, ax_cr = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            report_metrics, annot=True, cmap="Blues", fmt=".2f", linewidths=1, cbar=True, vmin=0, vmax=1, ax=ax_cr
        )
        style_mpl_dark(fig_cr, ax_cr)
        ax_cr.set_title("Classification Report Heatmap", fontsize=14, fontweight="bold")
        ax_cr.set_ylabel("Classes", fontsize=12, fontweight="bold")
        ax_cr.set_xlabel("Metrics", fontsize=12, fontweight="bold")
        fig_cr.tight_layout()
        st.pyplot(fig_cr)
        conclusion_box("classification_report", CONCLUSIONS["classification_report"])

    with col2:
        st.subheader("Slope Graph: Actual vs Predicted Class Counts")
        actual_counts = pd.Series(y_test).value_counts().sort_index()
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        classes = sorted(list(set(y_test) | set(y_pred)))

        fig_sl, ax_sl = plt.subplots(figsize=(8, 4))
        style_mpl_dark(fig_sl, ax_sl)
        x1, x2 = 0, 1
        colors_sl = ["#2ecc71", "#e74c3c", "#3498db", "#f1c40f"]

        for idx, cls in enumerate(classes):
            y1v = actual_counts.get(cls, 0)
            y2v = pred_counts.get(cls, 0)
            color = colors_sl[idx % len(colors_sl)]
            ax_sl.plot([x1, x2], [y1v, y2v], marker="o", markersize=8, linewidth=3, color=color, label=f"Class {cls}")
            ax_sl.text(x1 - 0.03, y1v, f"Class {cls}: {y1v}", ha="right", va="center", fontweight="bold", color=color)
            ax_sl.text(x2 + 0.03, y2v, f"Class {cls}: {y2v}", ha="left", va="center", fontweight="bold", color=color)

        ax_sl.set_xticks([x1, x2])
        ax_sl.set_xticklabels(["Actual (y_test)", "Predicted (y_pred)"], fontsize=12, fontweight="bold")
        ax_sl.set_xlim(-0.35, 1.35)
        ax_sl.axvspan(0.5, 1.35, color="grey", alpha=0.1, zorder=0)
        ax_sl.set_title("Slope Graph: Actual vs Predicted Class Counts", fontsize=14, fontweight="bold")
        ax_sl.spines["top"].set_visible(False)
        ax_sl.spines["right"].set_visible(False)
        ax_sl.spines["bottom"].set_visible(False)
        ax_sl.grid(axis="y", linestyle="--", alpha=0.2, color="white")
        fig_sl.tight_layout()
        st.pyplot(fig_sl)
        conclusion_box("slope_graph", CONCLUSIONS["slope_graph"])