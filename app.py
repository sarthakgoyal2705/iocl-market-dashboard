import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# --- Config ---
st.set_page_config(page_title="IOCL Market Dashboard", page_icon="🛢️", layout="wide")

st.title("🛢️ IOCL Market Dashboard")
st.markdown("Track USD/INR exchange rates and Crude Oil prices over time. **All prices are shown in Indian Rupees (₹) and times in IST.**")

# --- Tickers & Units ---
SYMBOLS = {
    "USD/INR": "INR=X",
    "Brent Crude": "BZ=F",
    "WTI Crude": "CL=F"
}

UNITS = {
    "USD/INR": "₹ per $",
    "Brent Crude": "₹ per Barrel",
    "WTI Crude": "₹ per Barrel"
}

# --- Sidebar Controls ---
st.sidebar.header("Controls")

# Auto-refresh every 10 seconds (10000 milliseconds)
st_autorefresh(interval=10000, limit=None, key="auto_refresh")

if st.sidebar.button("🔄 Force Refresh Data"):
    st.cache_data.clear()
    st.rerun()

selected_assets = st.sidebar.multiselect(
    "Select Assets to Track:",
    options=list(SYMBOLS.keys()),
    default=list(SYMBOLS.keys())
)

st.sidebar.subheader("Time Settings")
time_ranges = [
    "Last 5 Mins", 
    "Last 10 Mins", 
    "Last 30 Mins", 
    "Last 1 Hour", 
    "Last 1 Day", 
    "Last 5 Days", 
    "Last 1 Month", 
    "Last 6 Months", 
    "Last 1 Year",
    "Custom Time Window"
]
range_option = st.sidebar.selectbox("Select Time Range:", time_ranges, index=4)

yf_start = None
yf_end = None
yf_period = None
yf_interval = None

if range_option == "Custom Time Window":
    st.sidebar.markdown("*Note: 1-minute intraday data is only available for the last 7 days.*")
    max_date = datetime.today().date()
    min_date = max_date - timedelta(days=6)
    custom_date = st.sidebar.date_input("Select Date:", max_date, min_value=min_date, max_value=max_date)
    custom_start_time = st.sidebar.time_input("Start Time (IST):", datetime.strptime("10:00", "%H:%M").time())
    custom_end_time = st.sidebar.time_input("End Time (IST):", datetime.strptime("12:00", "%H:%M").time())
    
    yf_start = custom_date
    yf_end = custom_date + timedelta(days=1)
    yf_interval = "1m"
elif range_option in ["Last 5 Mins", "Last 10 Mins", "Last 30 Mins", "Last 1 Hour"]:
    yf_period = "1d"
    yf_interval = "1m"
elif range_option == "Last 1 Day":
    yf_period = "1d"
    yf_interval = "5m"
elif range_option == "Last 5 Days":
    yf_period = "5d"
    yf_interval = "15m"
elif range_option == "Last 1 Month":
    yf_period = "1mo"
    yf_interval = "1d"
elif range_option == "Last 6 Months":
    yf_period = "6mo"
    yf_interval = "1d"
elif range_option == "Last 1 Year":
    yf_period = "1y"
    yf_interval = "1d"
else:
    yf_period = "1mo"
    yf_interval = "1d"

# --- Price Alerts Sidebar ---
st.sidebar.markdown("---")
with st.sidebar.expander("🔔 Price Alerts"):
    alert_active = st.checkbox("Enable Alert")
    alert_asset = st.selectbox("Asset for Alert:", list(SYMBOLS.keys()))
    alert_condition = st.selectbox("Condition:", ["Drops Below", "Rises Above"])
    alert_price = st.number_input(f"Target Price (₹)", value=6000.0, step=100.0)


# --- Data Fetching ---
@st.cache_data(ttl=10)
def fetch_data(tickers, period, interval, start=None, end=None):
    fetch_tickers = tickers.copy()
    if "USD/INR" not in fetch_tickers:
        fetch_tickers["USD/INR"] = SYMBOLS["USD/INR"]
        
    data_dict = {}
    
    for name, ticker in fetch_tickers.items():
        try:
            if period:
                df = yf.download(ticker, period=period, interval=interval, progress=False)
            else:
                df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
                
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data_dict[name] = df[['Close']].copy()
                data_dict[name].rename(columns={'Close': name}, inplace=True)
        except Exception as e:
            st.error(f"Error fetching data for {name}: {e}")
            
    combined_df = pd.DataFrame()
    for name, df in data_dict.items():
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = combined_df.join(df, how='outer')
            
    if not combined_df.empty:
        combined_df.ffill(inplace=True)
        if combined_df.index.tz is None:
             combined_df.index = combined_df.index.tz_localize('UTC')
        try:
            combined_df.index = combined_df.index.tz_convert(ZoneInfo('Asia/Kolkata'))
        except Exception:
            pass
            
        if 'USD/INR' in combined_df.columns:
            usd_inr_series = combined_df['USD/INR']
            if 'Brent Crude' in combined_df.columns:
                combined_df['Brent Crude'] = combined_df['Brent Crude'] * usd_inr_series
            if 'WTI Crude' in combined_df.columns:
                combined_df['WTI Crude'] = combined_df['WTI Crude'] * usd_inr_series

    current_prices = {}
    daily_data = {}
    for name, ticker in fetch_tickers.items():
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="5d")
            if not hist.empty:
                daily_data[name] = hist['Close']
        except Exception:
            pass
            
    daily_df = pd.DataFrame(daily_data).ffill()
    if not daily_df.empty and 'USD/INR' in daily_df.columns:
        if 'Brent Crude' in daily_df.columns:
            daily_df['Brent Crude'] = daily_df['Brent Crude'] * daily_df['USD/INR']
        if 'WTI Crude' in daily_df.columns:
            daily_df['WTI Crude'] = daily_df['WTI Crude'] * daily_df['USD/INR']
            
        for name in fetch_tickers.keys():
            if name in daily_df.columns:
                series = daily_df[name].dropna()
                if len(series) >= 2:
                    current_val = float(series.iloc[-1])
                    prev_val = float(series.iloc[-2])
                    pct_change = ((current_val - prev_val) / prev_val) * 100 if prev_val else 0.0
                    current_prices[name] = {"price": current_val, "change": pct_change}
                elif len(series) == 1:
                    current_val = float(series.iloc[-1])
                    current_prices[name] = {"price": current_val, "change": 0.0}

    return combined_df, current_prices

if not selected_assets:
    st.info("Please select at least one asset from the sidebar.")
else:
    with st.spinner("Fetching market data..."):
        selected_tickers = {k: SYMBOLS[k] for k in selected_assets}
        historical_data, current_data = fetch_data(selected_tickers, yf_period, yf_interval, yf_start, yf_end)
        
    if not current_data:
        st.warning("No data found for the selected assets.")
    else:
        if not historical_data.empty:
            last_timestamp = historical_data.index.max()
            if range_option == "Custom Time Window":
                tz_ist = ZoneInfo('Asia/Kolkata')
                start_dt = datetime.combine(custom_date, custom_start_time).replace(tzinfo=tz_ist)
                end_dt = datetime.combine(custom_date, custom_end_time).replace(tzinfo=tz_ist)
                historical_data = historical_data[
                    (historical_data.index >= start_dt) & 
                    (historical_data.index <= end_dt)
                ]
            elif range_option == "Last 5 Mins":
                cutoff = last_timestamp - timedelta(minutes=5)
                historical_data = historical_data[historical_data.index >= cutoff]
            elif range_option == "Last 10 Mins":
                cutoff = last_timestamp - timedelta(minutes=10)
                historical_data = historical_data[historical_data.index >= cutoff]
            elif range_option == "Last 30 Mins":
                cutoff = last_timestamp - timedelta(minutes=30)
                historical_data = historical_data[historical_data.index >= cutoff]
            elif range_option == "Last 1 Hour":
                cutoff = last_timestamp - timedelta(minutes=60)
                historical_data = historical_data[historical_data.index >= cutoff]

        # --- Active Alerts Check ---
        if alert_active and alert_asset in current_data:
            curr_price = current_data[alert_asset]['price']
            if alert_condition == "Drops Below" and curr_price < alert_price:
                st.error(f"🚨 **ALERT!** {alert_asset} has dropped below ₹{alert_price:,.2f} (Current: ₹{curr_price:,.2f})", icon="🚨")
            elif alert_condition == "Rises Above" and curr_price > alert_price:
                st.warning(f"🚨 **ALERT!** {alert_asset} has risen above ₹{alert_price:,.2f} (Current: ₹{curr_price:,.2f})", icon="📈")

        # --- KPIs ---
        st.subheader("Current Market Rates")
        cols = st.columns(len(selected_assets))
        for idx, asset in enumerate(selected_assets):
            if asset in current_data:
                price = current_data[asset]['price']
                change = current_data[asset]['change']
                unit = UNITS[asset]
                
                cols[idx].metric(
                    label=f"{asset} ({unit})",
                    value=f"₹ {price:,.2f}",
                    delta=f"{change:.2f}%"
                )

        st.markdown("---")
        
        # --- TABS (Radio) ---
        current_tab = st.radio("Navigation", [
            "📈 Price Charts", 
            "⚖️ Spread Analysis", 
            "📊 Summary Statistics", 
            "🤖 AI Prediction",
            "📁 Raw Data"
        ], horizontal=True, label_visibility="collapsed")

        if current_tab == "📈 Price Charts":
            st.subheader(f"Price Trend: {range_option} (IST)")
            if range_option == "Custom Time Window" and not historical_data.empty:
                st.caption(f"Showing data for {custom_date} from {custom_start_time.strftime('%H:%M')} to {custom_end_time.strftime('%H:%M')} IST.")
            
            if not historical_data.empty:
                plot_data = historical_data.copy().reset_index()
                time_col = plot_data.columns[0]
                for asset in selected_assets:
                    if asset in plot_data.columns:
                        unit = UNITS[asset]
                        fig = px.line(plot_data, x=time_col, y=asset, 
                                      title=f"{asset} Historical Price",
                                      labels={asset: f"Price ({unit})", time_col: "Date/Time (IST)"})
                        fig.update_layout(hovermode="x unified", legend_title_text="")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No historical data available for the selected period.")

        elif current_tab == "⚖️ Spread Analysis":
            st.subheader("Brent vs. WTI Spread Analysis")
            if "Brent Crude" in historical_data.columns and "WTI Crude" in historical_data.columns:
                if "Brent Crude" in selected_assets and "WTI Crude" in selected_assets:
                    spread_df = historical_data.copy().reset_index()
                    time_col = spread_df.columns[0]
                    spread_df['Spread'] = spread_df['Brent Crude'] - spread_df['WTI Crude']
                    
                    fig_spread = px.area(spread_df, x=time_col, y="Spread",
                                         title="Premium of Brent Crude over WTI Crude (₹ per Barrel)",
                                         labels={"Spread": "Spread (₹)", time_col: "Date/Time (IST)"})
                    fig_spread.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_spread, use_container_width=True)
                    st.info("💡 **What is this?** The 'Spread' represents the price difference between Brent Crude and WTI Crude. A wider spread often indicates supply constraints in the Brent market or oversupply in the WTI market.")
                else:
                    st.warning("Please select both 'Brent Crude' and 'WTI Crude' in the sidebar to view the spread analysis.")
            else:
                 st.warning("Spread data is currently unavailable.")

        elif current_tab == "📊 Summary Statistics":
            st.subheader("Summary Statistics")
            if not historical_data[selected_assets].empty:
                stats_df = historical_data[selected_assets].describe().T
                if 'min' in stats_df.columns:
                    stats_df = stats_df[['min', 'max', 'mean', 'std']]
                    stats_df.rename(columns={'min': 'Lowest Price', 'max': 'Highest Price', 'mean': 'Average Price', 'std': 'Volatility (Std Dev)'}, inplace=True)
                    
                    for col in stats_df.columns:
                        stats_df[col] = stats_df[col].apply(lambda x: f"₹ {x:,.2f}" if pd.notnull(x) else "N/A")
                        
                    st.dataframe(stats_df, use_container_width=True)
            else:
                st.write("No data available to calculate statistics.")

        elif current_tab == "🤖 AI Prediction":
            st.subheader("🤖 AI Price Prediction (7-Day Forecast)")
            st.markdown("This feature uses a Machine Learning model (Polynomial Regression) trained on your selected historical data to forecast future trends.")
            
            ai_asset = st.selectbox("Select Asset to Predict:", selected_assets)
            
            if ai_asset and not historical_data.empty and ai_asset in historical_data.columns:
                df_pred = historical_data[[ai_asset]].copy().dropna()
                if len(df_pred) > 5:
                    # Convert DatetimeIndex to Epoch Seconds for precision in training
                    df_pred['Epoch'] = df_pred.index.astype('int64') / 10**9
                    X = df_pred[['Epoch']].values
                    y = df_pred[ai_asset].values
                    
                    poly = PolynomialFeatures(degree=2)
                    X_poly = poly.fit_transform(X)
                    model = LinearRegression()
                    model.fit(X_poly, y)
                    
                    df_pred['AI Trend'] = model.predict(X_poly)
                    
                    last_date = df_pred.index[-1]
                    # We predict the next 7 'steps' (days)
                    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
                    future_idx = pd.DatetimeIndex(future_dates)
                    future_epoch = future_idx.astype('int64') / 10**9
                    
                    future_poly = poly.transform(future_epoch.values.reshape(-1, 1))
                    future_preds = model.predict(future_poly)
                    
                    time_col_ai = 'Date/Time (IST)'
                    future_df = pd.DataFrame({
                        time_col_ai: future_dates,
                        'AI Forecast': future_preds
                    })
                    
                    plot_hist = df_pred.reset_index()
                    hist_time_col = plot_hist.columns[0]
                    
                    import plotly.graph_objects as go
                    fig_ai = go.Figure()
                    
                    fig_ai.add_trace(go.Scatter(x=plot_hist[hist_time_col], y=plot_hist[ai_asset], mode='lines', name=f'Historical {ai_asset}'))
                    fig_ai.add_trace(go.Scatter(x=plot_hist[hist_time_col], y=plot_hist['AI Trend'], mode='lines', name='AI Learned Trend', line=dict(dash='dot', color='orange')))
                    fig_ai.add_trace(go.Scatter(x=future_df[time_col_ai], y=future_df['AI Forecast'], mode='lines+markers', name='7-Day Forecast', line=dict(color='red', width=3)))
                    
                    fig_ai.update_layout(title=f"AI Forecast for {ai_asset}", hovermode="x unified", xaxis_title="Date/Time (IST)", yaxis_title=f"Price ({UNITS[ai_asset]})")
                    st.plotly_chart(fig_ai, use_container_width=True)
                    
                    st.write("**Forecasted Values:**")
                    future_df_disp = future_df.copy()
                    future_df_disp[time_col_ai] = future_df_disp[time_col_ai].apply(lambda x: x.strftime('%Y-%m-%d'))
                    future_df_disp['AI Forecast'] = future_df_disp['AI Forecast'].apply(lambda x: f"₹ {x:,.2f}")
                    st.dataframe(future_df_disp, hide_index=True)
                else:
                    st.warning("Not enough data points to train the AI model. Please select a larger time range.")

        elif current_tab == "📁 Raw Data":
            st.subheader("Raw Data (IST)")
            if not historical_data.empty:
                table_data = historical_data[selected_assets].sort_index(ascending=False).copy()
                if yf_interval in ["1m", "5m", "15m"]:
                    table_data.index = table_data.index.strftime('%Y-%m-%d %H:%M:%S IST')
                else:
                    table_data.index = table_data.index.strftime('%Y-%m-%d')
                    
                table_data.rename(columns={col: f"{col} ({UNITS[col]})" for col in table_data.columns}, inplace=True)
                
                st.dataframe(table_data, use_container_width=True)
                
                csv = table_data.to_csv()
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name="iocl_market_data_ist.csv",
                    mime="text/csv",
                )
