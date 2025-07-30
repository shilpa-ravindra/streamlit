import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📊 Market Profile (TPO) + Volume Profile")

# Get stock list
csv_files = [f for f in os.listdir('.') if f.endswith('_Historical_ONE_MINUTE_Data.csv')]
stock_names = [f.split('_')[0] for f in csv_files]

# --- Sidebar ---
st.sidebar.title("Controls")
selected_stock = st.sidebar.selectbox("Select Stock", stock_names)
selected_days = st.sidebar.selectbox("Select number of recent days", list(range(1, 11)))
bin_width = st.sidebar.slider("Bin Width", min_value=0.1, max_value=10.0, value=0.5, step=0.1)

file_name = f"{selected_stock}_Historical_ONE_MINUTE_Data.csv"

if not os.path.exists(file_name):
    st.error(f"File '{file_name}' not found.")
else:
    df = pd.read_csv(file_name, parse_dates=['DateTime'])
    df = df.sort_values('DateTime')

    # Validate columns
    required_columns = {'DateTime', 'Close', 'Volume'}
    if not required_columns.issubset(df.columns):
        st.error(f"CSV file must contain columns: {', '.join(required_columns)}")
        st.stop()

    df['Date'] = df['DateTime'].dt.date

    # Filter for recent N days
    recent_dates = df['Date'].drop_duplicates().sort_values().tail(selected_days)
    df = df[df['Date'].isin(recent_dates)]

    # Show the date range being analyzed
    st.markdown(f"📅 **Data Range Analyzed:** `{recent_dates.min()}` to `{recent_dates.max()}`")

    # Define price bins
    bins = np.arange(df['Close'].min(), df['Close'].max() + bin_width, bin_width)
    df['PriceBin'] = pd.cut(df['Close'], bins)

    # Market Profile (TPO)
    tpo_profile = df.groupby('PriceBin').size()
    volume_profile = df.groupby('PriceBin')['Volume'].sum()

    # Align bins
    all_bins = tpo_profile.index.union(volume_profile.index)
    bin_labels = [f"{b.left:.2f}-{b.right:.2f}" for b in all_bins]
    bin_centers = [b.mid for b in all_bins]
    tpo_counts = tpo_profile.reindex(all_bins, fill_value=0)
    volume_counts = volume_profile.reindex(all_bins, fill_value=0)

    # POCs
    poc_tpo = tpo_profile.idxmax().mid
    poc_vol = volume_profile.idxmax().mid

    # Value Area (70% TPO logic)
    total_tpo = tpo_counts.sum()
    target_tpo = total_tpo * 0.7
    cumulative = 0
    value_area_bins = []
    for idx, count in tpo_profile.sort_values(ascending=False).items():
        value_area_bins.append(idx)
        cumulative += count
        if cumulative >= target_tpo:
            break
    value_area_prices = [b.mid for b in value_area_bins]

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 10), sharey=True)

    # --- Market Profile Plot ---
    ax1.barh(bin_centers, tpo_counts, height=bin_width, color='lightgreen', edgecolor='black', label="TPO Count")
    ax1.axhline(poc_tpo, color='red', linestyle='--', linewidth=2, label=f'POC: {poc_tpo:.2f}')
    ax1.fill_betweenx(value_area_prices, 0, max(tpo_counts)*1.1, color='orange', alpha=0.3, label='Value Area (70%)')
    ax1.set_title("Market Profile (TPO)")
    ax1.set_xlabel("TPO Count")
    ax1.set_ylabel("Price")
    ax1.grid(True)
    ax1.legend()

    # --- Volume Profile Plot ---
    ax2.barh(bin_centers, volume_counts, height=bin_width, color='lightblue', edgecolor='black', label="Volume")
    ax2.axhline(poc_vol, color='blue', linestyle='--', linewidth=2, label=f'POC: {poc_vol:.2f}')
    ax2.set_title("Volume Profile")
    ax2.set_xlabel("Volume")
    ax2.set_yticks(bin_centers)
    ax2.set_yticklabels([f"{b:.2f}" for b in bin_centers])
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # --- Show Key Stats ---
    st.markdown("### 📌 Profile Stats")
    st.markdown(f"""
    - **POC (TPO-based):** `{poc_tpo:.2f}`
    - **POC (Volume-based):** `{poc_vol:.2f}`
    - **Value Area Range (TPO 70%):** `{min(value_area_prices):.2f}` to `{max(value_area_prices):.2f}`
    - **Total TPO Count:** `{total_tpo}`
    - **TPO Value Area Target (70%):** `{target_tpo:.0f}`
    """)
