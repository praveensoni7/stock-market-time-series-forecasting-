
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import importlib
from datetime import timedelta
from tensorflow.keras.models import load_model

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    import matplotlib.pyplot as plt
    PLOTLY_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ================================================================================

MODEL_PATH = r"script/gru_model.h5"
SCALER_PATH = r"script/scaler.pkl"

full_feature_cols = [
    "RSI_14", "MACD", "MACD_signal", "MACD_histogram", "OBV",
    "Stoch_%K", "Stoch_%D", "Williams_%R", 
    "BB_Width", "Daily_Return", "Volatility_20", "Close"
]

st.set_page_config(page_title="Stock Time-Series Forecast", layout="wide")


@st.cache_data
def load_data(path=r"dataset/P639 DATASET.csv"):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.set_index("Date").sort_index()
    return df


def calc_features(df):
    df = df.copy()

    # Moving averages
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # Exponential moving averages
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    df["BB_Std"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["BB_Std"]
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    # RSI
    rsi_w = 14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=rsi_w).mean()
    loss = (-delta).clip(lower=0).rolling(window=rsi_w).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["RSI_14"] = df["RSI_14"].fillna(50)

    # ATR / True Range
    df["High_Low"] = df["High"] - df["Low"]
    df["High_PrevClose"] = (df["High"] - df["Close"].shift(1)).abs()
    df["Low_PrevClose"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["True_Range"] = df[["High_Low", "High_PrevClose", "Low_PrevClose"]].max(axis=1)
    df["ATR_14"] = df["True_Range"].rolling(window=14).mean()

    # On-Balance Volume (OBV)
    obv = (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()
    df["OBV"] = obv

    # Stochastic oscillator (%K and %D)
    stoch_w = 14
    low_min = df["Low"].rolling(window=stoch_w).min()
    high_max = df["High"].rolling(window=stoch_w).max()
    df["Stoch_%K"] = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    df["Stoch_%D"] = df["Stoch_%K"].rolling(window=3).mean()

    # Williams %R
    wr_w = 14
    highest_high = df["High"].rolling(window=wr_w).max()
    lowest_low = df["Low"].rolling(window=wr_w).min()
    df["Williams_%R"] = -100 * (highest_high - df["Close"]) / (highest_high - lowest_low)

    # Price returns & volatility
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std() * np.sqrt(252)

    # Target diff used by model
    df["Close_diff"] = df["Close"].diff().diff()

    return df


def load_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    model = None
    scaler = None
    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        st.warning(f"Could not load keras model at {model_path}: {e}")
        model = None
    try:
        importlib.import_module("sklearn")
        scaler = joblib.load(scaler_path)
    except Exception as e:
        st.info(f"Scaler not loaded ({scaler_path}): {e} — a new scaler will be fitted if needed.")
        scaler = None
    return model, scaler


def create_scaled_array(df_feat, scaler, feature_cols):
    # ensure target `Close_diff` exists (compute if possible)
    if "Close_diff" not in df_feat.columns:
        if "Close" in df_feat.columns:
            df_feat = df_feat.copy()
            df_feat["Close_diff"] = df_feat["Close"].diff().diff()
            st.info("Computed missing 'Close_diff' from 'Close'.")
        else:
            raise KeyError("Required target column 'Close_diff' not found and 'Close' is unavailable to compute it.")

    arr_df = df_feat[feature_cols + ["Close_diff"]].dropna()
    if arr_df.shape[0] == 0:
        raise ValueError("No rows available after dropping NA for required features + Close_diff.")
    arr = arr_df.values

    scaler_used = scaler
    if scaler_used is not None:
        n_in = getattr(scaler_used, "n_features_in_", None)
        if n_in is not None and int(n_in) != arr.shape[1]:
            st.warning(f"Loaded scaler expects {n_in} features but current data has {arr.shape[1]}; a new scaler will be fitted.")
            scaler_used = None
        else:
            try:
                scaled = scaler_used.transform(arr)
            except Exception:
                st.info("Loaded scaler failed to transform current data; a new scaler will be fitted.")
                scaler_used = None

    if scaler_used is None:
        try:
            from sklearn.preprocessing import MinMaxScaler
        except Exception as e:
            raise ImportError("scikit-learn is required for fallback scaler. Install: pip install scikit-learn") from e
        fallback = MinMaxScaler()
        fallback.fit(arr)
        scaled = fallback.transform(arr)
        scaler_used = fallback
        st.info("Fitted new MinMaxScaler on current feature set.")

    return scaled, arr_df, scaler_used


def inverse_transform_preds(scaler, preds):
    # build temporary matrix matching scaler input width and place preds in last column
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in is None:
        n_in = preds.shape[1] + 1
    temp = np.zeros((len(preds), int(n_in)))
    temp[:, -1] = preds.flatten()
    inv = scaler.inverse_transform(temp)
    return inv[:, -1]


def evaluate_metrics(y_true, y_pred):
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error
    except Exception as e:
        raise ImportError("scikit-learn is required for evaluation metrics. Install: pip install scikit-learn") from e
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def align_feature_list(full_features, scaler, model):
    expected_total = None
    if scaler is not None:
        expected_total = getattr(scaler, "n_features_in_", None)
    if expected_total is None and model is not None:
        try:
            model_feat = model.input_shape[2]
            expected_total = int(model_feat) + 1
        except Exception:
            expected_total = None
    if expected_total is None:
        expected_total = len(full_features) + 1
    expected_features_no_target = int(expected_total) - 1
    if expected_features_no_target <= 0:
        raise ValueError("Invalid expected feature count inferred from artifacts.")
    if len(full_features) < expected_features_no_target:
        raise ValueError(
            f"Model/scaler expect {expected_features_no_target} features but available feature list has only {len(full_features)}. "
            "Either provide proper artifacts or retrain model with these features."
        )
    trimmed = full_features[:expected_features_no_target]
    if len(trimmed) != len(full_features):
        st.info(f"Trimming feature list from {len(full_features)} to {len(trimmed)} to match saved scaler/model.")
    return trimmed, expected_features_no_target


# ----------------- UI -----------------
st.title("Interactive Time-series Stock Forecast Dashboard")

df = load_data()
df = calc_features(df)

st.sidebar.header("Controls")
start_date = st.sidebar.date_input("Start date", df.index.min().date())
end_date = st.sidebar.date_input("End date", df.index.max().date())
if start_date > end_date:
    st.sidebar.error("Start must be <= End")

forecast_days = st.sidebar.number_input("Forecast horizon (days)", min_value=1, max_value=180, value=30)
lookback = st.sidebar.slider("Sequence lookback (for NN)", 10, 120, 60)
model_choice = st.sidebar.selectbox(
    "Deploy model for forecasting", ["GRU (saved)", "LSTM (saved)", "SARIMA (fit)", "ARIMA (fit)"]
)
evaluate_button = st.sidebar.button("Evaluate models on holdout")

# Historical display range (user can choose presets or custom start/end)
preset = st.sidebar.selectbox("Quick range (override Start date)", ["Custom", "1M", "3M", "6M", "YTD", "1Y", "5Y", "Max"], index=0)
end_dt = pd.to_datetime(end_date)
if preset == "Custom":
    plot_start = pd.to_datetime(start_date)
elif preset == "1M":
    plot_start = end_dt - pd.DateOffset(months=1)
elif preset == "3M":
    plot_start = end_dt - pd.DateOffset(months=3)
elif preset == "6M":
    plot_start = end_dt - pd.DateOffset(months=6)
elif preset == "YTD":
    plot_start = pd.to_datetime(f"{end_dt.year}-01-01")
elif preset == "1Y":
    plot_start = end_dt - pd.DateOffset(years=1)
elif preset == "5Y":
    plot_start = end_dt - pd.DateOffset(years=5)
else:
    plot_start = df.index.min()

# ensure plot_start is within available data
plot_start = max(plot_start, df.index.min())
display_df = df.loc[plot_start:end_dt]

st.subheader("Historical Close (Interactive)")
# allow overlaying indicators on the historical chart
overlay_options = [c for c in df.columns if c != "Close"]
overlay_selected = st.sidebar.multiselect("Overlay indicators on chart", options=overlay_options, default=[])

if PLOTLY_AVAILABLE:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=display_df.index, y=display_df["Close"], name="Close", line=dict(color="#1f77b4")))
    for col in overlay_selected:
        if col in display_df.columns:
            fig_hist.add_trace(go.Scatter(x=display_df.index, y=display_df[col], name=col, opacity=0.8))
    fig_hist.update_layout(title="Close Price", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    if overlay_selected:
        to_plot = display_df[["Close"] + [c for c in overlay_selected if c in display_df.columns]]
        st.line_chart(to_plot)
    else:
        st.line_chart(display_df["Close"])

model_loaded, scaler_loaded = load_artifacts()

# Allow user to filter which indicators to include (keep 'Close' mandatory)
mandatory_cols = ["Close"]
indicator_options = [c for c in full_feature_cols if c not in mandatory_cols]
selected_indicators = st.sidebar.multiselect("Select indicators to include", options=indicator_options, default=indicator_options)
chosen_full_features = list(selected_indicators) + mandatory_cols

# remove features the dataframe doesn't have and warn the user
available_features = [c for c in chosen_full_features if c in df.columns]
missing_features = [c for c in chosen_full_features if c not in df.columns]
if missing_features:
    st.warning(f"The following selected indicators are not present in the data and will be ignored: {missing_features}")
    if len(available_features) == 0:
        st.error("No selected indicators are available in the dataset. Adjust your selection.")
        st.stop()
chosen_full_features = available_features

try:
    feature_cols, expected_feat_count = align_feature_list(chosen_full_features, scaler_loaded, model_loaded)
except Exception as e:
    st.error(f"Feature alignment failed: {e}")
    st.stop()

st.info(f"Using {len(feature_cols)} features for preprocessing (expected by artifacts: {expected_feat_count}).")
st.write("Feature list used:", feature_cols)

try:
    scaled_all, valid_rows, scaler = create_scaled_array(df, scaler_loaded, feature_cols)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

if scaler_loaded is None and scaler is not None:
    if st.sidebar.checkbox("Save newly fitted scaler to disk (overwrite existing)"):
        try:
            joblib.dump(scaler, SCALER_PATH)
            st.sidebar.success(f"Scaler saved to {SCALER_PATH}")
        except Exception as e:
            st.sidebar.error(f"Saving scaler failed: {e}")

if evaluate_button:
    st.info("Evaluating models — this may take a while for ARIMA/SARIMA on long series.")
    df_clean = df[feature_cols + ["Close_diff"]].dropna()
    # use the original `Close` values aligned to the non-NA preprocessed rows
    target = df.loc[df_clean.index, "Close"]
    split = int(len(target) * 0.8)
    train_target = target.iloc[:split]
    test_target = target.iloc[split:]

    metrics = {}
    try:
        arima_m = ARIMA(train_target, order=(2, 1, 2)).fit()
        arima_preds = arima_m.forecast(steps=len(test_target))
        metrics["ARIMA"] = evaluate_metrics(test_target.values, arima_preds.values)
    except Exception:
        metrics["ARIMA"] = (np.nan, np.nan)

    try:
        sarima_m = SARIMAX(train_target, order=(2, 1, 2), seasonal_order=(2, 0, 2, 12)).fit(disp=False)
        sarima_preds = sarima_m.forecast(steps=len(test_target))
        metrics["SARIMA"] = evaluate_metrics(test_target.values, sarima_preds.values)
    except Exception:
        metrics["SARIMA"] = (np.nan, np.nan)

    if model_loaded is not None:
        try:
            lookback_model = model_loaded.input_shape[1]
            model_feat = model_loaded.input_shape[2]
            if model_feat != len(feature_cols):
                st.info(f"Model expects {model_feat} features; preprocessing produced {len(feature_cols)} — sequences will be sliced/truncated to match model.")
            arr = scaled_all
            X_seq, y_seq = [], []
            for i in range(len(arr) - lookback_model):
                X_seq.append(arr[i:i + lookback_model, :-1][:, :model_feat])
                y_seq.append(arr[i + lookback_model, -1])
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            if X_seq.size == 0:
                raise ValueError("Not enough sequence data after alignment.")
            split_seq = int(len(X_seq) * 0.8)
            X_test = X_seq[split_seq:]
            y_test = y_seq[split_seq:]
            preds_scaled = model_loaded.predict(X_test)
            preds_inv = inverse_transform_preds(scaler, preds_scaled.reshape(-1, 1))
            y_test_inv = inverse_transform_preds(scaler, y_test.reshape(-1, 1))
            metrics["LSTM_GRU"] = evaluate_metrics(y_test_inv, preds_inv)
        except Exception as e:
            st.warning(f"NN evaluation failed: {e}")
            metrics["LSTM_GRU"] = (np.nan, np.nan)
    else:
        metrics["LSTM_GRU"] = (np.nan, np.nan)

    metrics_df = pd.DataFrame(metrics, index=["RMSE", "MAE"]).T.reset_index().rename(columns={"index": "Model"})
    st.subheader("Model Evaluation on Holdout")
    st.dataframe(metrics_df)
    if PLOTLY_AVAILABLE:
        fig_metrics = px.bar(metrics_df.melt(id_vars="Model"), x="Model", y="value", color="variable", barmode="group",
                             title="RMSE and MAE by Model")
        st.plotly_chart(fig_metrics, use_container_width=True)
    else:
        st.bar_chart(metrics_df.set_index("Model"))

st.subheader("Generate Forecast")
if model_choice in ["GRU (saved)", "LSTM (saved)"] and model_loaded is None:
    st.warning("Selected NN model not loaded. Choose ARIMA/SARIMA or provide model artifact.")
else:
    valid_idx = valid_rows.index[valid_rows.index <= pd.to_datetime(end_date)]
    if len(valid_idx) == 0:
        st.error("No valid preprocessed rows up to selected end date.")
    else:
        last_valid_index = valid_idx[-1]
        pos = valid_rows.index.get_loc(last_valid_index)
        if pos + 1 < lookback:
            st.error("Not enough history up to selected end date for chosen lookback.")
        else:
            if model_choice in ["ARIMA (fit)", "SARIMA (fit)"]:
                train_target_full = df["Close"].loc[:last_valid_index]
                try:
                    if model_choice == "ARIMA (fit)":
                        m = ARIMA(train_target_full, order=(2, 1, 2)).fit()
                    else:
                        m = SARIMAX(train_target_full, order=(2, 1, 2), seasonal_order=(2, 0, 2, 12)).fit(disp=False)
                    preds = m.forecast(steps=forecast_days)
                    future_dates = pd.date_range(start=last_valid_index + timedelta(days=1), periods=forecast_days, freq="D")
                    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": preds.values}).set_index("Date")
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.loc[pd.to_datetime(start_date):last_valid_index].index,
                                                 y=df.loc[pd.to_datetime(start_date):last_valid_index, "Close"],
                                                 name="Historical"))
                        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Predicted_Close"], name="Forecast"))
                        fig.update_layout(title=f"{model_choice} Forecast", xaxis_title="Date", yaxis_title="Close")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.line_chart(pd.concat([df["Close"].loc[pd.to_datetime(start_date):last_valid_index], forecast_df["Predicted_Close"]]))
                    st.subheader("Forecast table")
                    st.dataframe(forecast_df)
                except Exception as e:
                    st.error(f"{model_choice} forecasting failed: {e}")
            else:
                if model_loaded is None:
                    st.error("Neural model not available for NN forecasting.")
                else:
                    lookback_model = model_loaded.input_shape[1]
                    model_feat = model_loaded.input_shape[2]
                    if len(scaled_all) < lookback_model:
                        st.error("Not enough preprocessed rows for model lookback.")
                    else:
                        pos_full = valid_rows.index.get_loc(last_valid_index)
                        seq_scaled = scaled_all[pos_full - lookback_model + 1: pos_full + 1, :-1]
                        # seq_scaled may be 1-D, 2-D (timesteps, features) or 3-D. Normalize to 2-D (timesteps, features).
                        if seq_scaled.ndim == 1:
                            seq_scaled = seq_scaled.reshape(lookback_model, -1)
                        if seq_scaled.ndim == 2:
                            n_feats = seq_scaled.shape[1]
                        else:
                            n_feats = seq_scaled.shape[2]

                        if n_feats != model_feat:
                            if n_feats < model_feat:
                                st.error(f"Model expects {model_feat} features but got {n_feats}; cannot proceed.")
                                st.stop()
                            # truncate extra features to match model
                            if seq_scaled.ndim == 2:
                                seq_scaled = seq_scaled[:, :model_feat]
                            else:
                                seq_scaled = seq_scaled[:, :, :model_feat]

                        # Ensure current_seq is 2-D (timesteps, features) for rolling and prediction
                        if seq_scaled.ndim == 3:
                            current_seq = seq_scaled.reshape(seq_scaled.shape[0], seq_scaled.shape[2])
                        else:
                            current_seq = seq_scaled.copy()

                        future_preds_scaled = []
                        for _ in range(forecast_days):
                            pred_scaled = model_loaded.predict(current_seq.reshape(1, lookback_model, current_seq.shape[1]))[0, 0]
                            future_preds_scaled.append(pred_scaled)
                            current_seq = np.roll(current_seq, -1, axis=0)
                            current_seq[-1, :] = current_seq[-2, :]

                        preds_scaled_arr = np.array(future_preds_scaled).reshape(-1, 1)
                        preds_diff_inv = inverse_transform_preds(scaler, preds_scaled_arr)
                        future_prices = np.cumsum(preds_diff_inv) + df.loc[last_valid_index, "Close"]
                        future_dates = pd.date_range(start=last_valid_index + timedelta(days=1), periods=forecast_days, freq="D")
                        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_prices}).set_index("Date")

                        if PLOTLY_AVAILABLE:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df.loc[pd.to_datetime(start_date):last_valid_index].index,
                                                     y=df.loc[pd.to_datetime(start_date):last_valid_index, "Close"],
                                                     name="Historical"))
                            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Predicted_Close"], name="Forecast"))
                            fig.update_layout(title=f"{model_choice} Forecast", xaxis_title="Date", yaxis_title="Close")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.line_chart(pd.concat([df["Close"].loc[pd.to_datetime(start_date):last_valid_index], forecast_df["Predicted_Close"]]))
                        st.subheader("Forecast table")
                        st.dataframe(forecast_df)
                        csv = forecast_df.reset_index().to_csv(index=False).encode("utf-8")
                        st.download_button("Download forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
