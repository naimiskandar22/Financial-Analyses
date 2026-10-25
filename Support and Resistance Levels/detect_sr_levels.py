#Load the libraries
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Find Support and Resistance Levels
def find_support_resistance(df, order=20):
    """Find support and resistance levels using local extrema."""
    close = df['Close'].values.flatten()
    
    # Find local maxima and minima
    local_max_idx = argrelextrema(close, np.greater, order=order)[0]
    local_min_idx = argrelextrema(close, np.less, order=order)[0]
    
    resistances = [(df.index[i], close[i]) for i in local_max_idx]
    supports = [(df.index[i], close[i]) for i in local_min_idx]
    
    return supports, resistances, local_min_idx, local_max_idx

def cluster_levels(levels, threshold_pct=0.02):
    """Cluster nearby price levels together."""
    if not levels:
        return []
    prices = sorted([l[1] for l in levels])
    clusters = [[prices[0]]]
    for p in prices[1:]:
        if (p - clusters[-1][-1]) / clusters[-1][-1] < threshold_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [np.mean(c) for c in clusters]

# ============================================================
# DAILY SUPPORT/RESISTANCE CONFIGURATION
# ============================================================

def assign_daily_sr(
    df,
    support_levels,
    resistance_levels,
    bounceback_days=5,
):
    """
    Assign a support and resistance level to each row in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex. Must contain 'High', 'Low', 'Close'.
    support_levels : list[float]
        Sorted list of horizontal support price levels (ascending).
    resistance_levels : list[float]
        Sorted list of horizontal resistance price levels (ascending).
    bounceback_days : int
        Number of trading days after a breakout within which the price must
        return inside the broken level for it to be considered a false breakout
        and the level restored. Default: 5.

    Returns
    -------
    pd.DataFrame
        Copy of df with two new columns:
        - 'Current_Resistance': the active resistance level for that day
        - 'Current_Support': the active support level for that day

    Algorithm / Key Variables
    -------------------------
    - s_levels / r_levels: sorted arrays of support/resistance prices.
    - cur_r_idx: index into r_levels pointing to the current resistance level.
    - cur_s_idx: index into s_levels pointing to the current support level.
    - r_breakout_start: the row index when High first exceeded the current
      resistance, or None if no active breakout. Used to track how many
      days have elapsed since the breakout.
    - s_breakout_start: same concept for support (Low dropped below).
    - Bounce-back check (resistance): on each day during an active breakout,
      if High < current resistance level, the breakout is a false alarm
      and the resistance is restored (r_breakout_start reset to None).
      If bounceback_days elapse without a return, the breakout is confirmed
      and cur_r_idx advances to the next higher resistance level.
    - Bounce-back check (support): symmetric logic — if Low > current support
      within the window, support is restored; otherwise cur_s_idx drops to
      the next lower support level.
    """
    # Sort levels ascending for orderly traversal
    s_levels = np.sort(support_levels)
    r_levels = np.sort(resistance_levels)

    close = df['Close'].values.flatten()
    high = df['High'].values.flatten()
    low = df['Low'].values.flatten()
    n = len(df)

    # Output arrays
    daily_resistance = np.full(n, np.nan)
    daily_support = np.full(n, np.nan)

    # --- Initialize: find the nearest resistance >= first close ---
    # cur_r_idx points to the current active resistance level in r_levels
    cur_r_idx = np.searchsorted(r_levels, close[0], side='left')
    if cur_r_idx >= len(r_levels):
        cur_r_idx = len(r_levels) - 1  # cap at highest known resistance

    # --- Initialize: find the nearest support <= first close ---
    # cur_s_idx points to the current active support level in s_levels
    cur_s_idx = np.searchsorted(s_levels, close[0], side='right') - 1
    if cur_s_idx < 0:
        cur_s_idx = 0  # cap at lowest known support

    # Breakout tracking: None means no active breakout
    # When set to an integer, it's the row index where the breakout started
    r_breakout_start = None  # tracks resistance breakout start day
    s_breakout_start = None  # tracks support breakdown start day

    for i in range(n):
        cur_r = r_levels[cur_r_idx]
        cur_s = s_levels[cur_s_idx]

        # ===================== RESISTANCE LOGIC =====================
        if r_breakout_start is None:
            # No active breakout — check if today's High breaks resistance
            if high[i] > cur_r:
                # Breakout detected: High exceeded the current resistance.
                # Start the bounceback countdown.
                r_breakout_start = i
        else:
            # Active breakout in progress
            days_since = i - r_breakout_start
            if high[i] <= cur_r:
                # BOUNCE-BACK: price (High) has returned below resistance.
                # This was a false breakout — restore the resistance level.
                r_breakout_start = None
            elif days_since >= bounceback_days:
                # CONFIRMED BREAKOUT: price stayed above resistance for
                # bounceback_days. Advance to the next higher resistance.
                r_breakout_start = None
                if cur_r_idx < len(r_levels) - 1:
                    cur_r_idx += 1

        # ===================== SUPPORT LOGIC ========================
        if s_breakout_start is None:
            # No active breakdown — check if today's Low breaks support
            if low[i] < cur_s:
                # Breakdown detected: Low dropped below current support.
                # Start the bounceback countdown.
                s_breakout_start = i
        else:
            # Active breakdown in progress
            days_since = i - s_breakout_start
            if low[i] >= cur_s:
                # BOUNCE-BACK: price (Low) has recovered above support.
                # This was a false breakdown — restore the support level.
                s_breakout_start = None
            elif days_since >= bounceback_days:
                # CONFIRMED BREAKDOWN: price stayed below support for
                # bounceback_days. Drop to the next lower support level.
                s_breakout_start = None
                if cur_s_idx > 0:
                    cur_s_idx -= 1

        # Record the current levels for this day
        daily_resistance[i] = r_levels[cur_r_idx]
        daily_support[i] = s_levels[cur_s_idx]

    # Attach results to a copy of the dataframe
    result = df.copy()
    result['Current_Resistance'] = daily_resistance
    result['Current_Support'] = daily_support
    return result

def assign_daily_resistance(df, resistance_levels, bounceback_days=5):
    """
    Assign the active resistance level to each row in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex. Must contain 'High', 'Close'.
    resistance_levels : list[float]
        List of horizontal resistance price levels.
    bounceback_days : int
        Number of trading days after a breakout within which the High must
        return below the broken resistance level for it to be considered a
        false breakout and the level restored. Default: 5.

    Returns
    -------
    np.ndarray
        Array of length len(df) with the active resistance for each day.

    Algorithm
    ---------
    - r_levels: sorted ascending array of resistance prices.
    - cur_r_idx: index into r_levels for the current active resistance.
    - On each day (when no active breakout):
        1. RE-EVALUATE: find the nearest resistance >= Close. This ensures
           resistance tracks downward when price drops significantly (e.g.
           the old resistance at $150 becomes irrelevant when price is at $100).
        2. Check if High > current resistance → start breakout timer.
    - During a breakout:
        * If High drops back <= resistance within bounceback_days → false
          breakout, restore the level.
        * If bounceback_days elapse with High still above → confirmed
          breakout, advance cur_r_idx to next higher level.
    """
    r_levels = np.sort(resistance_levels)
    close = df['Close'].values.flatten()
    high = df['High'].values.flatten()
    n = len(df)
    daily_resistance = np.full(n, np.nan)

    # Initialize: nearest resistance >= first close
    cur_r_idx = np.searchsorted(r_levels, close[0], side='left')
    if cur_r_idx >= len(r_levels):
        cur_r_idx = len(r_levels) - 1

    r_breakout_start = None  # row index where breakout started, or None

    for i in range(n):
        if r_breakout_start is None:
            # RE-EVALUATE: snap to the nearest resistance >= current Close.
            # This handles the case where price has fallen well below the
            # current resistance — we should track a closer (lower) one.
            ideal_idx = np.searchsorted(r_levels, close[i], side='left')
            if ideal_idx >= len(r_levels):
                ideal_idx = len(r_levels) - 1
            cur_r_idx = ideal_idx

            # Check for breakout: High exceeded current resistance
            if high[i] > r_levels[cur_r_idx]:
                r_breakout_start = i
        else:
            days_since = i - r_breakout_start
            if high[i] <= r_levels[cur_r_idx]:
                # BOUNCE-BACK: High returned below resistance → false breakout
                r_breakout_start = None
            elif days_since >= bounceback_days:
                # CONFIRMED BREAKOUT: advance to next higher resistance
                r_breakout_start = None
                if cur_r_idx < len(r_levels) - 1:
                    cur_r_idx += 1

        daily_resistance[i] = r_levels[cur_r_idx]

    return daily_resistance

def assign_daily_support(df, support_levels, bounceback_days=5):
    """
    Assign the active support level to each row in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex. Must contain 'Low', 'Close'.
    support_levels : list[float]
        List of horizontal support price levels.
    bounceback_days : int
        Number of trading days after a breakdown within which the Low must
        return above the broken support level for it to be considered a
        false breakdown and the level restored. Default: 5.

    Returns
    -------
    np.ndarray
        Array of length len(df) with the active support for each day.

    Algorithm
    ---------
    - s_levels: sorted ascending array of support prices.
    - cur_s_idx: index into s_levels for the current active support.
    - On each day (when no active breakdown):
        1. RE-EVALUATE: find the nearest support <= Close. This is the key
           fix — as price rises, support must track upward to the highest
           support level that is still below the current price. Without this,
           support would remain stuck at the initial low level forever.
        2. Check if Low < current support → start breakdown timer.
    - During a breakdown:
        * If Low recovers back >= support within bounceback_days → false
          breakdown, restore the level.
        * If bounceback_days elapse with Low still below → confirmed
          breakdown, drop cur_s_idx to the next lower level.
    """
    s_levels = np.sort(support_levels)
    close = df['Close'].values.flatten()
    low = df['Low'].values.flatten()
    n = len(df)
    daily_support = np.full(n, np.nan)

    # Initialize: nearest support <= first close
    cur_s_idx = np.searchsorted(s_levels, close[0], side='right') - 1
    if cur_s_idx < 0:
        cur_s_idx = 0

    s_breakout_start = None  # row index where breakdown started, or None

    for i in range(n):
        if s_breakout_start is None:
            # RE-EVALUATE: snap to the nearest support <= current Close.
            # This ensures support tracks UPWARD as price rises. For example,
            # if s_levels = [90, 110, 130] and close = 125, the active support
            # should be 110, not 90. searchsorted(side='right') - 1 gives the
            # largest level that is <= close.
            ideal_idx = np.searchsorted(s_levels, close[i], side='right') - 1
            if ideal_idx < 0:
                ideal_idx = 0
            cur_s_idx = ideal_idx

            # Check for breakdown: Low dropped below current support
            if low[i] < s_levels[cur_s_idx]:
                s_breakout_start = i
        else:
            days_since = i - s_breakout_start
            if low[i] >= s_levels[cur_s_idx]:
                # BOUNCE-BACK: Low recovered above support → false breakdown
                s_breakout_start = None
            elif days_since >= bounceback_days:
                # CONFIRMED BREAKDOWN: drop to next lower support
                s_breakout_start = None
                if cur_s_idx > 0:
                    cur_s_idx -= 1

        daily_support[i] = s_levels[cur_s_idx]

    return daily_support

def SetSRLevels(df, sr_threshold_pct = 0.03):
    supports, resistances, local_max_idx, local_min_idx = find_support_resistance(df, order=15)
    support_levels = cluster_levels(supports, threshold_pct=sr_threshold_pct)
    resistance_levels = cluster_levels(resistances, threshold_pct=sr_threshold_pct)
    # BOUNCEBACK_DAYS: number of trading days after a breakout within which
    #   the price must return inside the broken level for it to be restored.
    #   Higher = more forgiving (allows longer false breakouts).
    #   Lower = stricter (only very quick reversals count as bounce-backs).
    BOUNCEBACK_DAYS = 10

    # --- Run daily S/R assignment using separate functions ---
    df['Current_Resistance'] = assign_daily_resistance(df, resistance_levels, bounceback_days=BOUNCEBACK_DAYS)
    df['Current_Support'] = assign_daily_support(df, support_levels, bounceback_days=BOUNCEBACK_DAYS)

    return df