# Full combined Streamlit app with Supabase Auth, DB integration, CSV upload,
# leaderboard, goals, alerts, i18n, matplotlib charts, and GPT recommendations.

import os
import io
import datetime
import sqlite3
from typing import Dict, Any
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-4o-mini",
    input="Hello!",
)

print(response.output[0].content[0].text)

import streamlit as st
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Carbon Footprint Calculator — Full", page_icon="🌍", layout="wide")

EMISSION_FACTORS = {
    "Car (Petrol)": 0.192,
    "Car (Diesel)": 0.171,
    "Motorbike": 0.103,
    "Matatu/Bus": 0.105,
    "Bicycle/Walking": 0.0
}
ELECTRICITY_FACTOR = 0.18
LPG_FACTOR = 3.0
TABLE_NAME = "daily_emissions"

# Load environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------- SUPABASE / SQLITE --------------------

def init_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    if not SUPABASE_AVAILABLE:
        return None
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return client
    except Exception as e:
        st.error(f"Supabase init error: {e}")
        return None

supabase = init_supabase()

# Local fallback
sqlite_conn = sqlite3.connect("emissions.db", check_same_thread=False)
cur = sqlite_conn.cursor()
cur.execute(f"""
CREATE TABLE IF NOT EXISTS daily_emissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    alias TEXT,
    date TEXT,
    transport_mode TEXT,
    distance REAL,
    electricity REAL,
    lpg REAL,
    transport_emission REAL,
    electricity_emission REAL,
    lpg_emission REAL,
    total_emission REAL,
    notes TEXT
);
""")
cur.execute(f"""
CREATE TABLE IF NOT EXISTS user_goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    weekly_target REAL
);
""")
cur.execute(f"""
CREATE TABLE IF NOT EXISTS leaderboard_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    alias TEXT
);
""")
sqlite_conn.commit()

# -------------------- HELPERS --------------------

def compute_emissions(distance_km, transport_mode, electricity_kwh, lpg_kg):
    tf = EMISSION_FACTORS.get(transport_mode, 0.0)
    t_e = float(distance_km) * float(tf)
    e_e = float(electricity_kwh) * float(ELECTRICITY_FACTOR)
    l_e = float(lpg_kg) * float(LPG_FACTOR)
    total = t_e + e_e + l_e
    return {"transport_emission": t_e, "electricity_emission": e_e, "lpg_emission": l_e, "total_emission": total}

# DB insert functions both Supabase and local

def insert_local(record: Dict[str, Any]):
    cur = sqlite_conn.cursor()
    cur.execute("""
        INSERT INTO daily_emissions (user_id, alias, date, transport_mode, distance, electricity, lpg,
            transport_emission, electricity_emission, lpg_emission, total_emission, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record.get('user_id'), record.get('alias'), record.get('date'), record.get('transport_mode'), record.get('distance'),
        record.get('electricity'), record.get('lpg'), record.get('transport_emission'), record.get('electricity_emission'),
        record.get('lpg_emission'), record.get('total_emission'), record.get('notes')
    ))
    sqlite_conn.commit()

def fetch_all_local_for_user(user_id=None):
    if user_id:
        df = pd.read_sql_query("SELECT * FROM daily_emissions WHERE user_id=? ORDER BY date ASC", sqlite_conn, params=(user_id,))
    else:
        df = pd.read_sql_query("SELECT * FROM daily_emissions ORDER BY date ASC", sqlite_conn)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date']).dt.date
    return df

# Supabase analogs (best-effort)

def insert_supabase(record: Dict[str, Any]):
    try:
        supabase.table('daily_emissions').insert(record).execute()
        return True
    except Exception as e:
        st.error(f"Supabase insert error: {e}")
        return False

# -------------------- AUTH UI --------------------

def supabase_sign_in_ui():
    st.sidebar.markdown("### Account")
    if not supabase:
        st.sidebar.info("Supabase not configured: using local-only mode (no auth). Set SUPABASE_URL & SUPABASE_KEY to enable auth.")
        return None
    # Simple email auth using Magic Link
    if 'user' not in st.session_state:
        st.session_state['user'] = None
    if st.session_state['user'] is None:
        email = st.sidebar.text_input("Email for sign in (magic link)")
        if st.sidebar.button("Send Magic Link") and email:
            try:
                res = supabase.auth.sign_in_with_email(email=email)
                st.sidebar.success("Check your email for a magic link. After signing in, refresh this page.")
            except Exception as e:
                st.sidebar.error(f"Auth error: {e}")
        # Show manual 'Set as guest'
        if st.sidebar.button("Continue as guest (local only)"):
            st.session_state['user'] = {"id": f"guest-{os.getpid()}"}
            st.session_state['user_id'] = st.session_state['user']['id']
    else:
        st.sidebar.write(f"Signed in as: {st.session_state['user'].get('email', st.session_state['user'].get('id'))}")
        if st.sidebar.button("Sign out"):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            st.session_state['user'] = None

# NOTE: supabase-py may not have instant client-side session helpers. On Streamlit Cloud you can use
# Supabase's JavaScript SDK for a better auth flow. This code uses a best-effort Python-only approach.

# -------------------- PAGES --------------------

def page_home():
    st.title("🌍 Carbon Footprint Calculator — Full")
    st.write("Track, compare, and reduce your daily carbon emissions. Supports user accounts via Supabase.")

def page_enter_data():
    st.header("Enter or upload daily data")
    user_id = st.session_state.get('user_id') if 'user_id' in st.session_state else None
    alias = st.text_input("Display alias for leaderboard (optional)")
    col1, col2, col3 = st.columns([3,2,2])
    date = st.date_input("Date", value=datetime.date.today())
    with col1:
        distance = st.number_input("Distance (km or miles depending on unit below)", min_value=0.0, value=0.0)
        transport_mode = st.selectbox("Transport mode", list(EMISSION_FACTORS.keys()))
    with col2:
        electricity = st.number_input("Electricity (kWh)", min_value=0.0, value=0.0)
        lpg = st.number_input("LPG used (kg)", min_value=0.0, value=0.0)
    notes = st.text_area("Notes")

    if st.button("Save entry"):
        em = compute_emissions(distance, transport_mode, electricity, lpg)
        record = {
            'user_id': user_id,
            'alias': alias,
            'date': date.isoformat(),
            'transport_mode': transport_mode,
            'distance': float(distance),
            'electricity': float(electricity),
            'lpg': float(lpg),
            'transport_emission': em['transport_emission'],
            'electricity_emission': em['electricity_emission'],
            'lpg_emission': em['lpg_emission'],
            'total_emission': em['total_emission'],
            'notes': notes
        }
        if supabase:
            try:
                insert_supabase(record)
            except Exception:
                insert_local(record)
        else:
            insert_local(record)
        st.success(f"Saved — {em['total_emission']:.2f} kg CO₂")

    # CSV Upload
    st.markdown("---")
    st.markdown("### Import CSV (columns: date, distance, transport_mode, electricity, lpg, alias(optional), notes(optional))")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        # Validate and insert each row
        required = ['date','distance','transport_mode','electricity','lpg']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            count = 0
            for _, row in df.iterrows():
                try:
                    date_str = pd.to_datetime(row['date']).date().isoformat()
                    em = compute_emissions(row['distance'], row['transport_mode'], row['electricity'], row['lpg'])
                    rec = {
                        'user_id': user_id,
                        'alias': row.get('alias', alias),
                        'date': date_str,
                        'transport_mode': row['transport_mode'],
                        'distance': float(row['distance']),
                        'electricity': float(row['electricity']),
                        'lpg': float(row['lpg']),
                        'transport_emission': em['transport_emission'],
                        'electricity_emission': em['electricity_emission'],
                        'lpg_emission': em['lpg_emission'],
                        'total_emission': em['total_emission'],
                        'notes': row.get('notes','')
                    }
                    insert_local(rec)
                    count += 1
                except Exception as e:
                    st.warning(f"Failed to import row: {e}")
            st.success(f"Imported {count} rows into your history")


def page_history():
    st.header("History & Charts")
    user_id = st.session_state.get('user_id') if 'user_id' in st.session_state else None
    df = fetch_all_local_for_user(user_id)
    if df.empty:
        st.info("No data yet")
        return
    st.dataframe(df[['date','alias','distance','electricity','lpg','total_emission']].sort_values('date',ascending=False))

  
    df["date"] = pd.to_datetime(df["date"])

    # --- GROUP BY MONTH ---
    df_monthly = df.groupby(df["date"].dt.to_period("M")).sum().reset_index()
    df_monthly["date"] = df_monthly["date"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_monthly["date"], df_monthly["co2_total"], marker="o")
    ax.set_title("Monthly Total CO₂ Emissions")
    ax.set_xlabel("Month")
    ax.set_ylabel("kg CO₂")

    plt.xticks(rotation=45)
    st.pyplot(fig)


    # Breakdown stacked bars
    fig.autofmt_xdate()
    ax.tick_params(axis='x', labelrotation=45)
    plt.tight_layout()

    last = df.tail(14)
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.bar(last['date'], last['transport_emission'], label='Transport')
    ax2.bar(last['date'], last['electricity_emission'], bottom=last['transport_emission'], label='Electricity')
    bottoms = last['transport_emission'] + last['electricity_emission']
    ax2.bar(last['date'], last['lpg_emission'], bottom=bottoms, label='LPG')
    ax2.legend()
    ax2.set_xticklabels(last['date'], rotation=45)
    st.pyplot(fig2)
    st.subheader("Download Your Data")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
         label="📥 Download History CSV",
         data=csv,
         file_name="co2_history.csv",
         mime="text/csv",
     )


def page_goals_and_alerts():
    st.header("Goals & Alerts")
    user_id = st.session_state.get('user_id') if 'user_id' in st.session_state else None
    if not user_id:
        st.info("Set a weekly target — use guest or sign in to persist it.")
    current = None
    c = sqlite_conn.cursor()
    if user_id:
        c.execute("SELECT weekly_target FROM user_goals WHERE user_id=?", (user_id,))
        r = c.fetchone()
        if r:
            current = r[0]
    target = st.number_input("Weekly emissions target (kg CO2)", min_value=0.0, value=current or 20.0)
    if st.button("Save target"):
        if current is None:
            c.execute("INSERT INTO user_goals (user_id, weekly_target) VALUES (?,?)", (user_id, float(target)))
        else:
            c.execute("UPDATE user_goals SET weekly_target=? WHERE user_id=?", (float(target), user_id))
        sqlite_conn.commit()
        st.success("Saved goal")

    # Check weekly sum
    today = datetime.date.today()
    start_week = today - datetime.timedelta(days=today.weekday())
    df = fetch_all_local_for_user(user_id)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date']).dt.date
        week_df = df[(df['date'] >= start_week) & (df['date'] <= today)]
        weekly_total = week_df['total_emission'].sum()
        st.metric("This week's total (kg CO2)", f"{weekly_total:.2f}")
        if weekly_total > target:
            st.error("⚠️ You have exceeded your weekly target — consider reducing non-essential travel or electricity use this week.")
        else:
            st.success("👍 You're within your weekly target so far")


def page_leaderboard():
    st.header("Public Leaderboard (anonymous aliases)")
    # Produce leaderboard by weekly average reduction: simplest metric — weekly total ascending
    df_all = fetch_all_local_for_user(None)
    if df_all.empty:
        st.info("No data yet")
        return
    today = datetime.date.today()
    start_week = today - datetime.timedelta(days=7)
    df_all['date'] = pd.to_datetime(df_all['date']).dt.date
    week = df_all[df_all['date'] >= start_week]
    # aggregate by alias or user_id
    agg = week.groupby('alias').agg({'total_emission':'sum'}).reset_index()
    agg = agg.sort_values('total_emission')
    # anonymize blanks
    agg['alias'] = agg['alias'].fillna('Anonymous')
    st.table(agg.head(10).rename(columns={'total_emission':'weekly_total_kgCO2'}))

    st.markdown("**How leaderboard works**: People choose a display alias (optional) when saving entries. Top performers are lowest weekly totals. To make it fair, it only shows users with at least one entry this week.")


def page_insights():
    st.subheader("Quick Stats")
    total_saved = df["co2_total"].sum()
    avg_per_entry = df["co2_total"].mean()
    col1, col2 = st.columns(2)
    with col1:
      st.metric("Total (saved entries) kg CO₂", f"{total_saved:.2f}")
    with col2:
      st.metric("Average per entry (kg CO₂)", f"{avg_per_entry:.2f}")

    st.header("AI Recommendations (GPT)")
    if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        st.warning("OpenAI not configured. Set OPENAI_API_KEY to enable personalized recommendations.")
        return
    df = fetch_all_local_for_user(st.session_state.get('user_id'))
    if df.empty:
        st.info("No data to analyze — add entries first")
        return
    if st.button("Get GPT tips"):
        openai.api_key = OPENAI_API_KEY
        # compose prompt
        last = df.tail(7)
        summary_lines = []
        for _, r in last.iterrows():
            summary_lines.append(f"{r['date']}: {r['total_emission']:.2f} kg")
        prompt = "You are a sustainability assistant. Given recent daily CO2 totals:\n" + "\n".join(summary_lines) + "\nProvide 10 concise, actionable tips tailored to a user in Kenya to reduce transport, electricity, and cooking emissions."
   try:
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        input=prompt,
        max_output_tokens=300,
    )

    content = response.output_text   # much simpler!
    st.markdown(content)

   except Exception as e:
      st.error(f"AI Error: {e}")

# -------------------- EMBEDDABLE WIDGET SNIPPET --------------------

WIDGET_HTML = '''
<!-- Embeddable Carbon Widget: iframe snippet -->
<div id="carbon-widget">
  <iframe src="{APP_URL}/?embed=1" width="360" height="640" style="border:1px solid #eee;border-radius:8px"></iframe>
</div>
'''

# -------------------- MAIN APP LAYOUT --------------------

def main():
    st.sidebar.title("Navigation")
    supabase_sign_in_ui()
    pages = {
        'Home': page_home,
        'Enter Data': page_enter_data,
        'History': page_history,
        'Goals & Alerts': page_goals_and_alerts,
        'Leaderboard': page_leaderboard,
        'Insights': page_insights
    }
    choice = st.sidebar.radio("Go to", list(pages.keys()))
    pages[choice]()

if __name__ == '__main__':

    main()


