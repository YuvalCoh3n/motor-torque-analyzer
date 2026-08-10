import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
import io
import plotly.io as pio
from datetime import datetime, timedelta

# הגדרת דף
st.set_page_config(layout="wide", page_title="Motor Torque Analyzer")

st.title("📊 מנתח מומנט: פריצה מול עבודה יציבה")

uploaded_file = st.file_uploader("📁 העלה קובץ CSV", type="csv")

if uploaded_file is not None:
    # --- 1. קריאת נתונים וזיהוי כותרות אוטומטי ---
    content = uploaded_file.getvalue().decode("utf-8").splitlines()
    start_line = 0
    for i, line in enumerate(content):
        if line.startswith("X,") or "X," in line:
            start_line = i
            break
    
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, skiprows=start_line)
    df.columns = [col.strip() for col in df.columns]
    df = df.rename(columns={'X': 'Time_ms', '#03.002': 'Speed_RPM', '#04.002': 'Current_A'})

    for col in ['Time_ms', 'Speed_RPM', 'Current_A']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Time_ms', 'Speed_RPM', 'Current_A'])

    # --- 2. עיבוד והחלקה ---
    df['Torque_raw'] = -df['Current_A'] * 4.8
    df['Speed_RPM_fixed'] = -df['Speed_RPM']

    window = 51
    if len(df) < window: window = len(df) if len(df) % 2 != 0 else len(df) - 1
    
    if window > 3:
        df['Torque_smoothed'] = savgol_filter(df['Torque_raw'], window, polyorder=3)
        df['Speed_smoothed'] = savgol_filter(df['Speed_RPM_fixed'], window, polyorder=3)
    else:
        df['Torque_smoothed'], df['Speed_smoothed'] = df['Torque_raw'], df['Speed_RPM_fixed']

    # --- 3. זיהוי מקטעים וחישובים דינמיים ---
    moving_mask = df['Speed_smoothed'] > 5
    sections_indices = []
    start_idx = None

    for i in range(len(df)):
        if df['Speed_smoothed'].iloc[i] > 5 and start_idx is None:
            start_idx = i
        elif df['Speed_smoothed'].iloc[i] <= 5 and start_idx is not None:
            if i - start_idx > 10:
                sections_indices.append((start_idx, i))
            start_idx = None

    breakaway_peaks = []
    steady_state_means = []
    all_sections_data = []

    for start, end in sections_indices:
        sec_torque = df['Torque_smoothed'].iloc[start:end]
        sec_speed = df['Speed_smoothed'].iloc[start:end]
        breakaway_peaks.append(max(sec_torque))
        
        max_speed_in_sec = max(sec_speed)
        steady_mask = sec_speed > (max_speed_in_sec * 0.9)
        if steady_mask.any():
            steady_state_means.append(sec_torque[steady_mask].mean())
        
        all_sections_data.append(sec_torque.values)

    avg_breakaway = np.mean(breakaway_peaks) if breakaway_peaks else 0
    avg_steady = np.mean(steady_state_means) if steady_state_means else 0

    # --- 4. חישוב זמן מקומי (ישראל - UTC+2) ---
    # השרת בדרך כלל רץ על UTC, נוסיף 2 שעות לזמן המקומי
    local_time = datetime.utcnow() + timedelta(hours=2)
    date_str = local_time.strftime("%d.%m.%y")
    time_str = local_time.strftime("%H%M")
    display_time = local_time.strftime("%d.%m.%y %H:%M")

    # --- 5. תצוגת מדדים ---
    st.subheader(f"ניתוח הרצה: {display_time}")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("מומנט פריצה ממוצע", f"{avg_breakaway:.2f} Nm")
    col_b.metric("מומנט עבודה יציב", f"{avg_steady:.2f} Nm")
    col_c.metric("מספר מחזורים", len(breakaway_peaks))

    # --- 6. יצירת גרפים ---
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Scatter(x=df['Time_ms'], y=df['Torque_smoothed'], name='Torque [Nm]'), secondary_y=False)
    fig1.add_trace(go.Scatter(x=df['Time_ms'], y=df['Speed_smoothed'], name='Speed [RPM]', line=dict(color='green', dash='dot')), secondary_y=True)

    fig2 = go.Figure()
    if all_sections_data:
        min_len = min(len(s) for s in all_sections_data)
        section_array = np.array([s[:min_len] for s in all_sections_data])
        mean_curve = section_array.mean(axis=0)
        for s in section_array:
            fig2.add_trace(go.Scatter(y=s, mode='lines', opacity=0.2, line=dict(color='orange'), showlegend=False))
        fig2.add_trace(go.Scatter(y=mean_curve, name='ממוצע מחזורים', line=dict(color='black', width=3)))
        fig2.add_hline(y=avg_breakaway, line_dash="dash", line_color="red", annotation_text="Breakaway")
        fig2.add_hline(y=avg_steady, line_dash="dot", line_color="blue", annotation_text="Steady State")

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

    # --- 7. הורדת דוח עם שם קובץ מתוקן ---
    st.divider()
    
    # פורמט שם קובץ: 02.02.26-1216 - 8.60NM.html
    dynamic_filename = f"{date_str}-{time_str} - {avg_breakaway:.2f}NM.html"
    
    html_report = f"""
    <html dir='rtl'>
    <head><meta charset='utf-8'></head>
    <body style='font-family: sans-serif; padding: 20px;'>
        <h1>דוח בדיקת מנוע - {display_time}</h1>
        <h2 style='color: red;'>מומנט פריצה ממוצע: {avg_breakaway:.2f} Nm</h2>
        <h2 style='color: blue;'>מומנט עבודה יציב: {avg_steady:.2f} Nm</h2>
        <hr>
        {pio.to_html(fig1, full_html=False, include_plotlyjs='cdn')}
        <br>
        {pio.to_html(fig2, full_html=False, include_plotlyjs=False)}
    </body>
    </html>
    """

    st.download_button(
        label="📥 הורד דוח גרפים אינטראקטיבי",
        data=html_report,
        file_name=dynamic_filename,
        mime="text/html"
    )
