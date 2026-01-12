import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import re
import pdfkit
from jinja2 import Template
from datetime import datetime
import tempfile
import os

st.set_page_config(page_title="AutoInsight AI", layout="wide")

st.markdown("""
<style>
body { background-color:#f5f7ff; }
.title { font-size:46px;font-weight:800;color:#4f46e5; }
.subtitle { font-size:18px;color:#64748b;margin-bottom:20px; }
.section { font-size:30px;font-weight:700;color:#475569;margin-top:40px; }
.card { background:#ffffff;padding:30px;border-radius:20px;
box-shadow:0 12px 28px rgba(0,0,0,0.06);margin-top:20px;color:#475569; }
.subhead { font-size:20px;font-weight:700;color:#6366f1;margin-top:22px; }
p { font-size:15px;line-height:1.7; }
ul { margin-top:14px;padding-left:20px; }
li { margin-bottom:12px;font-size:15px;line-height:1.6; }
.badge { display:inline-block;background:#eef2ff;color:#4f46e5;
padding:6px 14px;border-radius:999px;font-size:13px;font-weight:600;margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📊 AutoInsight AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Business Understanding & Exploratory Data Analysis</div>', unsafe_allow_html=True)
st.markdown("---")

def llama(prompt):
    for _ in range(3):
        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_ctx": 8192}
                },
                timeout=None
            )
            return r.json().get("response", "")
        except:
            continue
    return ""

def clean(text):
    return re.sub(r"\n{2,}", "\n", text).strip()

uploaded = st.file_uploader("Upload any CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.markdown('<div class="section">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    rows, cols = df.shape
    missing = int(df.isnull().sum().sum())
    dup = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", rows)
    c2.metric("Columns", cols)
    c3.metric("Missing", missing)
    c4.metric("Duplicates", dup)

    if st.button("Generate Full Professional Analysis"):
        understanding_prompt = f"""
You are a senior business analyst.

Return content strictly in this structure:

OVERVIEW:
2–3 lines explaining what the dataset represents.

DOMAIN:
Industry / business domain.

WHAT THIS DATA ENABLES:
At least 4 bullet points.

LIMITATIONS:
At least 3 bullet points.

PROFIT IMPROVEMENT ACTIONS:
At least 10 bullet points.

Dataset columns:
{df.columns.tolist()}

Sample data:
{df.head(5).to_string()}
"""
        raw = clean(llama(understanding_prompt))

        sections = {
            "OVERVIEW": "",
            "DOMAIN": "",
            "WHAT THIS DATA ENABLES": [],
            "LIMITATIONS": [],
            "PROFIT IMPROVEMENT ACTIONS": []
        }

        current = None
        for line in raw.split("\n"):
            line = line.strip()
            if line.endswith(":") and line[:-1] in sections:
                current = line[:-1]
                continue
            if current:
                if line.startswith(("-", "•", "*")):
                    sections[current].append(line.strip("-•* ").strip())
                else:
                    if isinstance(sections[current], list):
                        continue
                    sections[current] += " " + line

        st.markdown('<div class="section">Dataset Understanding</div>', unsafe_allow_html=True)

        st.markdown(f"""
<div class="card">
<span class="badge">Overview</span>
<p>{sections["OVERVIEW"]}</p>

<div class="subhead">🏭 Domain</div>
<p>{sections["DOMAIN"]}</p>

<div class="subhead">📊 What This Data Enables</div>
<ul>{''.join([f"<li>{x}</li>" for x in sections["WHAT THIS DATA ENABLES"]])}</ul>

<div class="subhead">⚠️ Limitations</div>
<ul>{''.join([f"<li>{x}</li>" for x in sections["LIMITATIONS"]])}</ul>

<div class="subhead">🚀 Profit Improvement Actions</div>
<ul>{''.join([f"<li>{x}</li>" for x in sections["PROFIT IMPROVEMENT ACTIONS"]])}</ul>
</div>
""", unsafe_allow_html=True)

        charts_html = []
        revenue_col = "total_price" if "total_price" in df.columns else df.select_dtypes(include=np.number).columns[-1]

        st.markdown('<div class="section">Visual Analysis</div>', unsafe_allow_html=True)

        if "pizza_category" in df.columns:
            fig = px.bar(df.groupby("pizza_category")[revenue_col].sum().reset_index(),
                         x="pizza_category", y=revenue_col, title="Revenue by Category")
            st.plotly_chart(fig, use_container_width=True)
            charts_html.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

        if "pizza_size" in df.columns:
            fig = px.pie(df, names="pizza_size", values=revenue_col, title="Revenue Share by Size")
            st.plotly_chart(fig, use_container_width=True)
            charts_html.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

        insight_prompt = f"""
Write 8–10 strong business insights with recommendations.
Avoid generic points.

Statistics:
{df.describe(include='all').fillna('').to_string()}
"""
        insights = [i.strip("-• ") for i in llama(insight_prompt).split("\n") if len(i.strip()) > 20]

        st.markdown('<div class="section">Key Insights</div>', unsafe_allow_html=True)
        for i in insights:
            st.write("•", i)

        template = Template("""
<html>
<body style="font-family:Arial;padding:40px;color:#475569">
<h1 style="color:#4f46e5">AutoInsight AI – Professional Business EDA Report</h1>
<p><b>Generated:</b> {{ date }}</p>

<h2>Dataset Understanding</h2>
<p>{{ overview }}</p>

<h3>Domain</h3>
<p>{{ domain }}</p>

<h3>What This Data Enables</h3>
<ul>{% for i in enables %}<li>{{ i }}</li>{% endfor %}</ul>

<h3>Limitations</h3>
<ul>{% for i in limits %}<li>{{ i }}</li>{% endfor %}</ul>

<h3>Profit Improvement Actions</h3>
<ul>{% for i in actions %}<li>{{ i }}</li>{% endfor %}</ul>

<h2>Data Quality</h2>
<p>Rows: {{ rows }} | Columns: {{ cols }}</p>
<p>Missing: {{ missing }} | Duplicates: {{ dup }}</p>

<h2>Visual Analysis</h2>
{% for c in charts %}{{ c | safe }}<br><br>{% endfor %}

<h2>Key Insights</h2>
<ul>{% for i in insights %}<li>{{ i }}</li>{% endfor %}</ul>
</body>
</html>
""")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            pdf_path = f.name

        html = template.render(
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            overview=sections["OVERVIEW"],
            domain=sections["DOMAIN"],
            enables=sections["WHAT THIS DATA ENABLES"],
            limits=sections["LIMITATIONS"],
            actions=sections["PROFIT IMPROVEMENT ACTIONS"],
            rows=rows, cols=cols, missing=missing, dup=dup,
            charts=charts_html, insights=insights
        )

        pdfkit.from_string(html, pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download Professional Report", f, file_name="AutoInsight_Report.pdf")

        os.remove(pdf_path)
