#Load the libraries
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Automated Triangle Pattern Detection
# ============================================================
# TRIANGLE DETECTION CONFIGURATION — Adjust these to tune sensitivity
# ============================================================

# --- Sliding Window Parameters ---
# window_sizes: list of rolling window lengths (in trading days) to scan.
#   Shorter windows (30-40) catch small, fast-forming triangles.
#   Longer windows (80-120) catch large, slow-forming triangles.
#   Using multiple sizes scans across different timeframes.
TRIANGLE_WINDOW_SIZES = [30, 45, 60, 80, 100]

# step: how many bars to advance the sliding window each iteration.
#   Smaller step = more overlap = more candidate patterns found (but slower).
TRIANGLE_STEP = 3

# --- Local Extrema Parameters ---
# extrema_order: number of bars on each side a point must be higher/lower than
#   to qualify as a local max/min. Lower = more extrema detected = more
#   trendline candidates. Higher = only major pivots.
TRIANGLE_EXTREMA_ORDER = 3

# min_touches: minimum number of local highs (or lows) required to fit a
#   trendline. Classic definition needs 2; using 2 is the most permissive.
TRIANGLE_MIN_TOUCHES = 2

# --- Trendline Quality ---
# min_r_squared: minimum R² for the linear fit of the upper/lower trendline.
#   1.0 = perfect fit, 0.0 = no fit. Lower values allow noisier trendlines.
#   Strict: 0.65+   Moderate: 0.40   Relaxed: 0.20
TRIANGLE_MIN_R_SQUARED = 0.20

# --- Slope Classification Thresholds ---
# These control how the normalized slope of each trendline is classified.
# Slopes are normalized as: slope * window_length / price_range_of_segment
# This makes them comparable across different price levels and timeframes.
#
# flat_threshold: if |normalized_slope| < this, the trendline is considered "flat".
#   A flat upper line + rising lower line => Ascending Triangle (bullish).
#   A flat lower line + falling upper line => Descending Triangle (bearish).
#   Relaxed: 0.5   Moderate: 0.3   Strict: 0.15
TRIANGLE_FLAT_THRESHOLD = 0.5

# converging_threshold: minimum |normalized_slope| for the non-flat trendline
#   to be considered "converging" toward the flat line.
#   For ascending: lower trendline slope must exceed this (positive).
#   For descending: upper trendline slope must be below -this (negative).
#   Relaxed: 0.08   Moderate: 0.15   Strict: 0.25
TRIANGLE_CONVERGING_THRESHOLD = 0.08

# symmetrical_min_slope: for symmetrical triangles, both trendlines must
#   converge. Upper must slope down more than this, lower must slope up more.
#   Relaxed: 0.05   Moderate: 0.12   Strict: 0.20
TRIANGLE_SYMMETRICAL_MIN_SLOPE = 0.05

# --- Deduplication ---
# dedup_overlap: minimum index distance between pattern start points to
#   consider them distinct. Patterns closer than this are deduplicated,
#   keeping the one with highest confidence.
TRIANGLE_DEDUP_OVERLAP = 15

def detect_triangles(
    df,
    window_sizes=None,
    step=3,
    order=3,
    min_touches=2,
    min_r_squared=0.20,
    flat_threshold=0.5,
    converging_threshold=0.08,
    symmetrical_min_slope=0.05,
):
    """
    Detect ascending, descending, and symmetrical triangle patterns.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    window_sizes : list[int]
        List of rolling window lengths (trading days) to scan. Multiple sizes
        allow detection across different timeframes. Default: [30,45,60,80,100].
    step : int
        How many bars to advance the window each iteration. Smaller = more
        overlap and more candidate patterns (slower). Default: 3.
    order : int
        The `order` parameter for scipy.signal.argrelextrema. A point must be
        the highest (or lowest) within `order` bars on each side. Lower values
        produce more local extrema, increasing sensitivity. Default: 3.
    min_touches : int
        Minimum number of local highs (or lows) needed to fit a trendline.
        Classic triangles need at least 2 touches per line. Default: 2.
    min_r_squared : float
        Minimum R² goodness-of-fit for each trendline's linear regression.
        Values near 0 allow noisy trendlines; values near 1 require near-perfect
        alignment of the extrema. Default: 0.20.
    flat_threshold : float
        Maximum |normalized_slope| for a trendline to be classified as "flat".
        Normalized slope = raw_slope * window / price_range, making it
        dimensionless and comparable across price levels.
        - Ascending: upper trendline is flat, lower slopes upward.
        - Descending: lower trendline is flat, upper slopes downward.
        Higher values are more permissive (allow slight slope as "flat").
        Default: 0.5.
    converging_threshold : float
        Minimum |normalized_slope| for the converging (non-flat) trendline.
        Ensures the non-flat line is actually moving toward the flat one.
        Lower values = more permissive. Default: 0.08.
    symmetrical_min_slope : float
        For symmetrical triangles: both lines must converge. Upper must have
        normalized_slope < -symmetrical_min_slope and lower must have
        normalized_slope > symmetrical_min_slope. Default: 0.05.

    Returns
    -------
    list[dict] : Detected patterns with type, date range, slopes, and confidence.

    Key Variables in the Algorithm
    ------------------------------
    - segment : the price slice within the current window
    - local_max_idx / local_min_idx : indices of swing highs / lows within the segment
    - upper_slope / lower_slope : raw slope ($/bar) from np.polyfit on the extrema
    - upper_r2 / lower_r2 : R² of the linear fit; measures how well extrema
      align on a straight line (1.0 = perfect)
    - price_range : max - min of the segment; used to normalize slopes
    - norm_upper / norm_lower : dimensionless slope = raw_slope * window / price_range
      Positive = rising, negative = falling, near-zero = flat
    """
    if window_sizes is None:
        window_sizes = [30, 45, 60, 80, 100]

    close = df['Close'].values.flatten()
    patterns = []

    for window in window_sizes:
        for start in range(0, len(close) - window, step):
            # Extract the price segment for this window position
            segment = close[start:start + window]

            # Find local swing highs and lows within the segment.
            # `order` controls how dominant a peak/trough must be.
            local_max_idx = argrelextrema(segment, np.greater, order=order)[0]
            local_min_idx = argrelextrema(segment, np.less, order=order)[0]

            # Need at least min_touches swing points to fit a meaningful trendline
            if len(local_max_idx) < min_touches or len(local_min_idx) < min_touches:
                continue

            # --- Fit linear trendlines via least-squares regression ---
            # Upper trendline: line through the swing highs
            # coeffs[0] = slope ($/bar), coeffs[1] = intercept ($)
            coeffs_upper = np.polyfit(local_max_idx, segment[local_max_idx], 1)
            upper_slope = coeffs_upper[0]

            # Lower trendline: line through the swing lows
            coeffs_lower = np.polyfit(local_min_idx, segment[local_min_idx], 1)
            lower_slope = coeffs_lower[0]

            # --- Compute R² (goodness of fit) for each trendline ---
            # R² = 1 - (sum of squared residuals) / (total sum of squares)
            # High R² means the swing points sit neatly on a line.
            upper_fit = np.polyval(coeffs_upper, local_max_idx)
            lower_fit = np.polyval(coeffs_lower, local_min_idx)

            ss_res_upper = np.sum((segment[local_max_idx] - upper_fit) ** 2)
            ss_tot_upper = np.var(segment[local_max_idx]) * len(local_max_idx) + 1e-10
            upper_r2 = 1 - ss_res_upper / ss_tot_upper

            ss_res_lower = np.sum((segment[local_min_idx] - lower_fit) ** 2)
            ss_tot_lower = np.var(segment[local_min_idx]) * len(local_min_idx) + 1e-10
            lower_r2 = 1 - ss_res_lower / ss_tot_lower

            # Skip if either trendline is a poor fit
            if upper_r2 < min_r_squared or lower_r2 < min_r_squared:
                continue

            # --- Normalize slopes to make them comparable across price levels ---
            # price_range: the total vertical span of the segment
            # norm_slope = raw_slope * window / price_range
            #   Interpretation: how much (as a fraction of price_range) the
            #   trendline moves over the full window.
            #   +1.0 means the line rises by the full price range over the window.
            #   0.0 means the line is perfectly flat.
            price_range = segment.max() - segment.min()
            if price_range == 0:
                continue
            norm_upper = upper_slope * window / price_range
            norm_lower = lower_slope * window / price_range

            # --- Classify the pattern based on trendline slopes ---
            pattern_type = None

            # ASCENDING TRIANGLE: flat (or nearly flat) resistance line on top,
            # rising support line from below. Typically bullish — buyers push
            # lows higher while price repeatedly tests a ceiling.
            if abs(norm_upper) < flat_threshold and norm_lower > converging_threshold:
                pattern_type = "Ascending Triangle"

            # DESCENDING TRIANGLE: falling resistance line on top, flat (or
            # nearly flat) support line below. Typically bearish — sellers push
            # highs lower while price tests a floor.
            elif norm_upper < -converging_threshold and abs(norm_lower) < flat_threshold:
                pattern_type = "Descending Triangle"

            # SYMMETRICAL TRIANGLE: both lines converge — upper slopes down,
            # lower slopes up, squeezing price into an apex. Neutral bias;
            # breakout direction determines the trade.
            elif norm_upper < -symmetrical_min_slope and norm_lower > symmetrical_min_slope:
                pattern_type = "Symmetrical Triangle"

            if pattern_type:
                patterns.append({
                    'type': pattern_type,
                    'window': window,
                    'start_idx': start,
                    'end_idx': start + window,
                    'start_date': df.iloc[start].name,
                    'end_date': df.iloc[min(start + window - 1, len(df) - 1)].name,
                    'upper_slope': upper_slope,
                    'lower_slope': lower_slope,
                    'norm_upper': norm_upper,
                    'norm_lower': norm_lower,
                    'confidence': (upper_r2 + lower_r2) / 2,
                    'local_max_idx': local_max_idx + start,
                    'local_min_idx': local_min_idx + start,
                })

    return patterns

# # --- Run detection with configurable parameters ---
# triangles = detect_triangles(
#     df,
#     window_sizes=TRIANGLE_WINDOW_SIZES,
#     step=TRIANGLE_STEP,
#     order=TRIANGLE_EXTREMA_ORDER,
#     min_touches=TRIANGLE_MIN_TOUCHES,
#     min_r_squared=TRIANGLE_MIN_R_SQUARED,
#     flat_threshold=TRIANGLE_FLAT_THRESHOLD,
#     converging_threshold=TRIANGLE_CONVERGING_THRESHOLD,
#     symmetrical_min_slope=TRIANGLE_SYMMETRICAL_MIN_SLOPE,
# )

# Remove duplicated/overlapping patterns
def get_unique_triangles(patterns):
    # --- Deduplicate overlapping patterns (keep highest confidence) ---
    unique_triangles = []
    for t in sorted(patterns, key=lambda x: -x['confidence']):
        if not any(abs(t['start_idx'] - u['start_idx']) < TRIANGLE_DEDUP_OVERLAP and
                t['type'] == u['type'] for u in unique_triangles):
            unique_triangles.append(t)

    return unique_triangles

# Display triangle pattern results
def display_triangle_pattern_results(triangles, unique_patterns):
    if not unique_patterns:
        print("No triangle patterns detected. Try lowering min_r_squared or adjusting slope thresholds.")
        return 0

    # --- Results ---
    print(f"Raw candidates: {len(triangles)}  |  After dedup: {len(unique_patterns)}\n")
    for i, t in enumerate(unique_patterns):
        print(f"  [{i+1}] {t['type']}  (window={t['window']}d)")
        print(f"      Period: {t['start_date'].strftime('%Y-%m-%d')} to {t['end_date'].strftime('%Y-%m-%d')}")
        print(f"      Confidence: {t['confidence']:.2%}")
        print(f"      Norm slopes — upper: {t['norm_upper']:.3f}  lower: {t['norm_lower']:.3f}")
        signal = "BULLISH" if "Ascending" in t['type'] else "BEARISH" if "Descending" in t['type'] else "NEUTRAL"
        print(f"      Signal: {signal}\n")

def plot_triangle_patterns(df, unique_patterns, max_patterns = 6):
    if not unique_patterns:
        print("No triangle patterns detected. Try lowering min_r_squared or adjusting slope thresholds.")
        return 0
    # Plot triangle patterns
    # --- Plot top patterns ---
    for i, t in enumerate(unique_patterns[:max_patterns]):

        seg = df.iloc[t['start_idx']:t['end_idx']]
        close_vals = df['Close'].values.flatten()

        # Plot candlestick chart
        fig = go.Figure(data=[go.Candlestick(x=seg.index,
                    open=seg['Open'],
                    high=seg['High'],
                    low=seg['Low'],
                    close=seg['Close'])])

        # Plot triangle resistance line 
        max_idx = t['local_max_idx']
        max_idx = max_idx[max_idx < len(close_vals)]
        coeffs_u = np.polyfit(max_idx - t['start_idx'], close_vals[max_idx], 1)
        # Plot triangle support line
        min_idx = t['local_min_idx']
        min_idx = min_idx[min_idx < len(close_vals)]
        coeffs_l = np.polyfit(min_idx - t['start_idx'], close_vals[min_idx], 1)

        # Plot support/resistance trendlines
        x_range = np.arange(0, t['end_idx'] - t['start_idx'])

        fig.add_scatter(x=seg.index, y=np.polyval(coeffs_u, x_range[:len(seg)]), mode="lines",
                        line=dict(color="#ff1100", width=2, dash='dash'),
                        name="Resistance")
        fig.add_scatter(x=seg.index, y=np.polyval(coeffs_l, x_range[:len(seg)]), mode="lines",
                        line=dict(color="#22ff00", width=2, dash='dash'),
                        name="Support")

        signal = "BULLISH" if "Ascending" in t['type'] else "BEARISH" if "Descending" in t['type'] else "NEUTRAL"
        # print(f"      Signal: {signal}\n")
        title = f"{t['type']} → Signal: {signal} {t['type']} (Confidence: {t['confidence']:.2f}, Window: {t['window']})"

        # ── Layout ───────────────────────────────────────────────────────────────
        fig.update_layout(
            template='plotly_dark',
            height=900,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            title_text=f'{title}'
        )
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price ($)')
        fig.show()