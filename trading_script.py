import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta, time, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.signal import argrelextrema
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# List of symbols to analyze
symbols = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "EURCHF=X", "AUDJPY=X", "CADJPY=X", "NZDJPY=X",
    "GBPCAD=X", "GBPCHF=X", "EURAUD=X", "EURCAD=X", "AUDCAD=X", "AUDCHF=X", "AUDNZD=X",
    "NZDCAD=X", "NZDCHF=X", "CADCHF=X", "EURNZD=X", "GBPAUD=X", "GBPNZD=X",
    "USDMXN=X", "USDZAR=X", "USDTRY=X", "USDSEK=X", "USDNOK=X", "USDDKK=X", "USDSGD=X",
    "USDHKD=X", "USDCNH=X", "USDPLN=X", "USDINR=X", "USDBRL=X", "USDCZK=X", "USDHUF=X", "USDILS=X",
    "GC=F", "SI=F", "PA=F", "PL=F",  # Gold, Silver, Palladium, Platinum
    "CL=F", "NG=F", "RB=F", "HO=F", "BZ=F",  # Crude Oil, Natural Gas, RBOB Gasoline, Heating Oil, Brent Crude
    "^GSPC", "^DJI", "^IXIC", "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI", "^BSESN", "^AXJO"  # Major indices
]

def fetch_and_process_data(symbol, start_date, end_date, intervals):
    data = {}
    for interval in intervals:
        try:
            df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            if not df.empty:
                data[interval] = process_data(df)
                print(f"Data fetched for {symbol} at interval {interval}")
            else:
                print(f"No data available for {symbol} at interval {interval}")
        except Exception as e:
            print(f"Error fetching data for {symbol} at interval {interval}: {e}")
    
    if not data:
        print(f"No data available for {symbol} at any interval")
        return None
    
    return data

def process_data(df):
    df = calculate_ict_indicators(df)
    return df

def calculate_ict_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Moving Averages
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # Fair Value Gap (FVG)
    df['FVG_up'] = (df['Low'].shift(2) > df['High'].shift(1)) & (df['Open'] < df['Low'].shift(2))
    df['FVG_down'] = (df['High'].shift(2) < df['Low'].shift(1)) & (df['Open'] > df['High'].shift(2))
    
    # Optimal Trade Entry (OTE)
    df['OTE_Long'] = (df['Low'] <= df['EMA_200']) & (df['Close'] > df['EMA_200'])
    df['OTE_Short'] = (df['High'] >= df['EMA_200']) & (df['Close'] < df['EMA_200'])
    
    # Institutional Candles
    df['Institutional_Candle'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low']) > 0.6) | ((df['High'] - df['Close']) / (df['High'] - df['Low']) > 0.6)

    # Liquidity Sweeps
    df['Liquidity_Sweep_Low'] = df['Low'] < df['Low'].rolling(20).min().shift()
    df['Liquidity_Sweep_High'] = df['High'] > df['High'].rolling(20).max().shift()

    # Breaker Blocks
    df['Breaker_Bullish'] = (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Low'] < df['Low'].shift(1))
    df['Breaker_Bearish'] = (df['Close'] < df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)) & (df['High'] > df['High'].shift(1))

    # Order Blocks
    df['Order_Block_Bullish'] = (df['Close'] < df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)) & (df['High'] > df['High'].shift(1))
    df['Order_Block_Bearish'] = (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Low'] < df['Low'].shift(1))

    # Breaker Blocks (alternative definition)
    df['Breaker_Block_Bullish'] = (df['Low'] < df['Low'].shift(1)) & (df['High'] > df['High'].shift(1)) & (df['Close'] > df['Open'])
    df['Breaker_Block_Bearish'] = (df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1)) & (df['Close'] < df['Open'])

    # Rejection Blocks
    df['Rejection_Block_Bullish'] = (df['Close'] < df['Open']) & (df['Low'] < df['Low'].shift(1)) & (df['High'] > df['High'].shift(1))
    df['Rejection_Block_Bearish'] = (df['Close'] > df['Open']) & (df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1))

    # RSI and Divergence Phantoms
    df['RSI'] = calculate_rsi(df['Close'])
    df['Divergence_Phantom_Bullish'] = (df['Close'] < df['Close'].shift(1)) & (df['RSI'] > df['RSI'].shift(1))
    df['Divergence_Phantom_Bearish'] = (df['Close'] > df['Close'].shift(1)) & (df['RSI'] < df['RSI'].shift(1))

    # Double Patterns
    df['Double_Bottom'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2)) & (df['Low'] > df['Low'].shift(1))
    df['Double_Top'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2)) & (df['High'] < df['High'].shift(1))

    # Swing High/Low
    df['Swing_High'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
    df['Swing_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))

    # Premium/Discount Zones
    df['Premium_Zone'] = df['Close'] > df['EMA_50'] * 1.02
    df['Discount_Zone'] = df['Close'] < df['EMA_50'] * 0.98

    # Equilibrium
    df['Equilibrium'] = (df['Close'] > df['EMA_50'] * 0.99) & (df['Close'] < df['EMA_50'] * 1.01)

    # Imbalance
    df['Imbalance_Bullish'] = df['Low'] > df['High'].shift(1)
    df['Imbalance_Bearish'] = df['High'] < df['Low'].shift(1)

    return df

def determine_overall_direction(data: Dict[str, pd.DataFrame]) -> str:
    df = data['1d']  # Use daily data for overall direction
    sma_50 = df['Close'].rolling(window=50).mean()
    sma_200 = df['Close'].rolling(window=200).mean()
    
    if sma_50.iloc[-1] > sma_200.iloc[-1] and df['Close'].iloc[-1] > sma_50.iloc[-1]:
        return "Bullish"
    elif sma_50.iloc[-1] < sma_200.iloc[-1] and df['Close'].iloc[-1] < sma_50.iloc[-1]:
        return "Bearish"
    else:
        return "Neutral"

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean().iloc[-1]

def calculate_fibonacci_levels(data, trend):
    if trend == 'bullish':
        swing_high = data['High'].max()
        swing_low = data['Low'].min()
    else:  # bearish
        swing_high = data['High'].max()
        swing_low = data['Low'].min()
    
    diff = swing_high - swing_low
    
    levels = {
        0: 0,
        0.236: 0.236,
        0.382: 0.382,
        0.5: 0.5,
        0.618: 0.618,
        0.786: 0.786,
        1: 1
    }
    
    if trend == 'bullish':
        fib_levels = {level: swing_low + diff * ratio for level, ratio in levels.items()}
    else:  # bearish
        fib_levels = {level: swing_high - diff * ratio for level, ratio in levels.items()}
    
    return fib_levels

def find_closest_fibonacci_level(price, fib_levels):
    closest_level = min(fib_levels, key=lambda x: abs(fib_levels[x] - price))
    return closest_level, fib_levels[closest_level]

def determine_trend(data, short_period=10, long_period=30):
    short_ema = data['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_period, adjust=False).mean()
    
    if short_ema.iloc[-1] > long_ema.iloc[-1] and short_ema.iloc[-2] <= long_ema.iloc[-2]:
        return 'Bullish'
    elif short_ema.iloc[-1] < long_ema.iloc[-1] and short_ema.iloc[-2] >= long_ema.iloc[-2]:
        return 'Bearish'
    elif short_ema.iloc[-1] > long_ema.iloc[-1]:
        return 'Bullish'
    else:
        return 'Bearish'

def calculate_fibonacci_levels(data, trend):
    if trend == 'Bullish':
        low = data['Low'].min()
        high = data['High'].max()
    else:
        low = data['High'].max()
        high = data['Low'].min()
    diff = high - low
    levels = {
        0: low,
        0.236: low + 0.236 * diff,
        0.382: low + 0.382 * diff,
        0.5: low + 0.5 * diff,
        0.618: low + 0.618 * diff,
        0.786: low + 0.786 * diff,
        1: high,
        1.618: high + 0.618 * diff,
        2.618: high + 1.618 * diff
    }
    return levels

def find_closest_fibonacci_level(current_price, fib_levels):
    closest_level = min(fib_levels, key=lambda x: abs(fib_levels[x] - current_price))
    return closest_level, fib_levels[closest_level]

def determine_kill_zone(current_time):
    hour = current_time.hour
    minute = current_time.minute
    day = current_time.weekday()
    
    if day < 5:  # Monday to Friday
        if (8 <= hour < 10) or (14 <= hour < 16):
            return "Active"
        elif hour == 10 and minute < 30:
            return "Active"
        elif hour == 16 and minute < 30:
            return "Active"
    
    return "Inactive"

def calculate_stop_loss(data, current_price, trend, atr_period=14, atr_multiplier=2):
    atr = data['TR'].rolling(window=atr_period).mean().iloc[-1]
    if trend == 'Bullish':
        return current_price - atr_multiplier * atr
    else:
        return current_price + atr_multiplier * atr

def calculate_take_profit(current_price, stop_loss, risk_reward_ratio=2):
    distance = abs(current_price - stop_loss)
    if current_price > stop_loss:
        return current_price + (distance * risk_reward_ratio)
    else:
        return current_price - (distance * risk_reward_ratio)

def predict_price_movement(data, model):
    # Prepare the features for the current candle
    current_features = prepare_ict_features(data.iloc[-1])
    
    prediction = model.predict(current_features.reshape(1, -1))
    return "Up" if prediction[0] == 1 else "Down"

def prepare_ict_features(candle):
    # Extract ICT-based features from the current candle
    ob_distance = calculate_ob_distance(candle)
    liq_level_distance = calculate_liq_level_distance(candle)
    fvg_presence = identify_fair_value_gaps(candle)
    institutional_candle = identify_institutional_candles(candle)
    
    return np.array([ob_distance, liq_level_distance, fvg_presence, institutional_candle])

def assess_risk_management(stop_loss, take_profit, current_price, account_balance, risk_percentage=2):
    risk = abs(current_price - stop_loss)
    reward = abs(take_profit - current_price)
    risk_reward_ratio = reward / risk if risk != 0 else 0
    
    max_risk_amount = account_balance * (risk_percentage / 100)
    position_size = max_risk_amount / risk
    
    if risk_reward_ratio >= 3:
        return "Excellent", position_size
    elif 2 <= risk_reward_ratio < 3:
        return "Good", position_size
    elif 1 <= risk_reward_ratio < 2:
        return "Moderate", position_size
    else:
        return "Poor", 0

def identify_order_blocks(df, lookback=10):
    ob_highs = []
    ob_lows = []
    if len(df) < lookback:
        return [], []  # Return empty lists if not enough data
    
    for i in range(len(df) - lookback):
        if df['High'].iloc[i] == df['High'].iloc[i:i+lookback].max():
            ob_highs.append((df.index[i], df['High'].iloc[i]))
        if df['Low'].iloc[i] == df['Low'].iloc[i:i+lookback].min():
            ob_lows.append((df.index[i], df['Low'].iloc[i]))
    return ob_highs, ob_lows

def identify_liquidity_levels(df, lookback=20):
    liq_highs = []
    liq_lows = []
    for i in range(len(df) - lookback):
        if df['High'].iloc[i] == df['High'].iloc[i:i+lookback].max():
            liq_highs.append((df.index[i], df['High'].iloc[i]))
        if df['Low'].iloc[i] == df['Low'].iloc[i:i+lookback].min():
            liq_lows.append((df.index[i], df['Low'].iloc[i]))
    return liq_highs, liq_lows

def identify_institutional_candles(df, threshold=1.5):
    atr = calculate_atr(df)
    institutional_candles = []
    for i in range(len(df)):
        candle_range = df['High'].iloc[i] - df['Low'].iloc[i]
        if candle_range > threshold * atr:
            institutional_candles.append((df.index[i], 'Bullish' if df['Close'].iloc[i] > df['Open'].iloc[i] else 'Bearish'))
    return institutional_candles

def determine_entry_signal(data, trend, fib_levels, kill_zone_info, order_blocks, liquidity_levels):
    current_price = data.iloc[-1]['Close']
    previous_price = data.iloc[-2]['Close']
    closest_fib, fib_price = find_closest_fibonacci_level(current_price, fib_levels)
    
    if kill_zone_info == "Active":
        if trend == 'Bullish':
            if current_price > previous_price and current_price > fib_price:
                if any(ob['Low'] < current_price < ob['High'] for _, ob in order_blocks[0].iterrows()):
                    if any(ll['Low'] < current_price for _, ll in liquidity_levels[1].iterrows()):
                        return 'BUY'
        elif trend == 'Bearish':
            if current_price < previous_price and current_price < fib_price:
                if any(ob['Low'] < current_price < ob['High'] for _, ob in order_blocks[1].iterrows()):
                    if any(current_price < ll['High'] for _, ll in liquidity_levels[0].iterrows()):
                        return 'SELL'
    
    return 'HOLD'

def analyze_kill_zone(df):
    current_time = df.index[-1].time()
    
    # Define all kill zones
    asian_open = time(23, 0)  # 11:00 PM UTC
    asian_close = time(8, 0)  # 8:00 AM UTC
    london_open = time(8, 0)  # 8:00 AM UTC
    london_close = time(16, 30)  # 4:30 PM UTC
    ny_open = time(13, 30)  # 1:30 PM UTC
    ny_close = time(22, 0)  # 10:00 PM UTC
    ny_close_session = time(21, 0)  # 9:00 PM UTC
    
    kill_zones = []
    
    # Check for ICT kill zones
    if asian_open <= current_time or current_time < asian_close:
        kill_zones.append("Asian")
    if london_open <= current_time < london_close:
        kill_zones.append("London")
    if ny_open <= current_time < ny_close:
        kill_zones.append("New York")
    if ny_close_session <= current_time < ny_close:
        kill_zones.append("New York Close")
    
    # Special cases for ICT concepts
    if london_open <= current_time < (london_open.replace(hour=london_open.hour + 1)):
        kill_zones.append("London Open")
    if ny_open <= current_time < (ny_open.replace(hour=ny_open.hour + 1)):
        kill_zones.append("New York Open")
    
    if len(kill_zones) > 1:
        return f"Overlapping Kill Zones: {' & '.join(kill_zones)}"
    elif len(kill_zones) == 1:
        return f"{kill_zones[0]} Kill Zone"
    else:
        return "Not in Kill Zone"

def is_in_kill_zone(df):
    kill_zone_info = analyze_kill_zone(df)
    return kill_zone_info != "Not in Kill Zone"

def identify_breaker_blocks(df, lookback=10):
    breaker_blocks = []
    for i in range(len(df) - lookback, len(df)):
        if df['High'].iloc[i] > df['High'].iloc[i-1:i].max() and df['Low'].iloc[i] < df['Low'].iloc[i-1:i].min():
            breaker_blocks.append((df.index[i], 'Bullish' if df['Close'].iloc[i] > df['Open'].iloc[i] else 'Bearish'))
    return breaker_blocks

def identify_order_blocks(df, lookback=10):
    ob_highs = []
    ob_lows = []
    if len(df) < lookback:
        return [], []  # Return empty lists if not enough data
    
    for i in range(len(df) - lookback):
        if df['High'].iloc[i] == df['High'].iloc[i:i+lookback].max():
            ob_highs.append((df.index[i], df['High'].iloc[i]))
        if df['Low'].iloc[i] == df['Low'].iloc[i:i+lookback].min():
            ob_lows.append((df.index[i], df['Low'].iloc[i]))
    return ob_highs, ob_lows

def identify_fair_value_gaps(df):
    fvg = []
    for i in range(1, len(df) - 1):
        if df['Low'].iloc[i] > df['High'].iloc[i-1] and df['Low'].iloc[i] > df['High'].iloc[i+1]:
            fvg.append((df.index[i], 'Bullish'))
        elif df['High'].iloc[i] < df['Low'].iloc[i-1] and df['High'].iloc[i] < df['Low'].iloc[i+1]:
            fvg.append((df.index[i], 'Bearish'))
    return fvg

def identify_liquidity_levels(df, lookback=20):
    liq_highs = []
    liq_lows = []
    for i in range(len(df) - lookback):
        if df['High'].iloc[i] == df['High'].iloc[i:i+lookback].max():
            liq_highs.append((df.index[i], df['High'].iloc[i]))
        if df['Low'].iloc[i] == df['Low'].iloc[i:i+lookback].min():
            liq_lows.append((df.index[i], df['Low'].iloc[i]))
    return liq_highs, liq_lows

def get_ml_prediction(model: RandomForestClassifier, scaler: StandardScaler, latest_data: pd.Series) -> float:
    features = [
        'EMA_21', 'OTE_Long', 'OTE_Short', 'FVG_up', 'FVG_down',
        'Institutional_Candle', 'Liquidity_Sweep_Low', 'Liquidity_Sweep_High',
        'Breaker_Bullish', 'Breaker_Bearish', 'Breaker_Block_Bullish',
        'Breaker_Block_Bearish', 'Rejection_Block_Bullish', 'Rejection_Block_Bearish',
        'Divergence_Phantom_Bullish', 'Divergence_Phantom_Bearish',
        'Double_Bottom', 'Double_Top', 'RSI'
    ]
    X = latest_data[features].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[0][1]  # Probability of price increase

def analyze_data(data: Dict[str, pd.DataFrame], ml_model: RandomForestClassifier, scaler: StandardScaler, ml_confidence_threshold: float) -> Dict:
    if not data:
        return None

    try:
        print("Starting analysis...")
        available_intervals = sorted(data.keys(), key=lambda x: {'1mo': 0, '1wk': 1, '1d': 2, '1h': 3}.get(x, 4), reverse=True)
        entry_interval = available_intervals[0]
        df = data[entry_interval]
        
        current_price = df['Close'].iloc[-1]
        overall_direction = determine_overall_direction(data)
        
        # Smart Money Concepts (SMC) Analysis
        smc_analysis = analyze_smc(df)
        
        # Order Flow Analysis
        order_flow = analyze_order_flow(df)
        
        # Liquidity Analysis
        liquidity_analysis = analyze_liquidity(df)
        
        # Market Structure Analysis
        market_structure = analyze_market_structure(df)
        
        # Institutional Candles
        institutional_candles = identify_institutional_candles(df)
        
        # Kill Zone Analysis
        kill_zone_info = analyze_kill_zone(df)
        in_kill_zone = is_in_kill_zone(df)
        
        # Entry Signal Generation
        entry_signal = generate_entry_signal(df, overall_direction, smc_analysis, order_flow, liquidity_analysis, market_structure)
        
        # Machine Learning Prediction
        ml_prediction = get_ml_prediction(ml_model, scaler, df.iloc[-1])
        
        # Adjust entry signal based on ML prediction
        entry_signal = adjust_signal_with_ml(entry_signal, ml_prediction, ml_confidence_threshold, overall_direction)
        
        # Risk Management
        atr = calculate_atr(df)
        entry_price = current_price
        stop_loss, take_profit = calculate_risk_levels(entry_price, atr, entry_signal)
        risk_reward_ratio = calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)
        risk_management = "Good" if risk_reward_ratio >= 1.5 else "Poor"
        
        # Fibonacci Levels
        trend = determine_trend(data['1d'])
        fib_levels = calculate_fibonacci_levels(data['1d'], trend)
        closest_fib_level, closest_fib_price = find_closest_fibonacci_level(current_price, fib_levels)
        
        # Compile analysis results
        analysis_results = {
            'current_price': current_price,
            'overall_direction': overall_direction,
            'entry_signal': entry_signal,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'ml_prediction': ml_prediction,
            'risk_management': risk_management,
            'closest_fib_level': closest_fib_level,
            'closest_fib_price': closest_fib_price,
            'kill_zone_info': kill_zone_info,
            'in_kill_zone': in_kill_zone,
            'smc_analysis': smc_analysis,
            'order_flow': order_flow,
            'liquidity_analysis': liquidity_analysis,
            'market_structure': market_structure,
            'institutional_candles': institutional_candles,
            'trend': trend
        }

        print("Analysis completed successfully")
        return analysis_results
    except Exception as e:
        print(f"Error in analyze_data: {e}")
        return None

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def identify_institutional_candles(df, threshold=1.5):
    atr = calculate_atr(df)
    institutional_candles = []
    for i in range(len(df)):
        candle_range = df['High'].iloc[i] - df['Low'].iloc[i]
        if candle_range > threshold * atr:
            institutional_candles.append((df.index[i], 'Bullish' if df['Close'].iloc[i] > df['Open'].iloc[i] else 'Bearish'))
    return institutional_candles

def analyze_smc(df):
    # Implement Smart Money Concepts analysis
    breaker_blocks = identify_breaker_blocks(df)
    order_blocks = identify_order_blocks(df)
    fair_value_gaps = identify_fair_value_gaps(df)
    return {
        'breaker_blocks': breaker_blocks,
        'order_blocks': order_blocks,
        'fair_value_gaps': fair_value_gaps
    }

def analyze_order_flow(df):
    # Implement order flow analysis
    # This could include volume analysis, delta, etc.
    return "Order flow analysis result"

def analyze_liquidity(df):
    # Implement liquidity analysis
    liquidity_levels = identify_liquidity_levels(df)
    return {
        'liquidity_levels': liquidity_levels
    }

def analyze_market_structure(df):
    # Implement market structure analysis
    # This could include identifying higher highs, lower lows, etc.
    return "Market structure analysis result"

def generate_entry_signal(df, overall_direction, smc_analysis, order_flow, liquidity_analysis, market_structure):
    # Implement logic to generate entry signal based on all analyses
    # This is a placeholder and should be implemented based on your specific strategy
    return "HOLD"

def adjust_signal_with_ml(entry_signal, ml_prediction, ml_confidence_threshold, overall_direction):
    if ml_prediction > ml_confidence_threshold:
        if entry_signal == "BUY":
            return "STRONG BUY"
        elif entry_signal == "HOLD" and overall_direction == "Bullish":
            return "BUY"
    elif ml_prediction < (1 - ml_confidence_threshold):
        if entry_signal == "SELL":
            return "STRONG SELL"
        elif entry_signal == "HOLD" and overall_direction == "Bearish":
            return "SELL"
    return entry_signal

def calculate_risk_levels(entry_price, atr, entry_signal):
    if entry_signal in ["BUY", "STRONG BUY"]:
        stop_loss = entry_price - (2 * atr)
        take_profit = entry_price + (3 * atr)
    elif entry_signal in ["SELL", "STRONG SELL"]:
        stop_loss = entry_price + (2 * atr)
        take_profit = entry_price - (3 * atr)
    else:
        stop_loss = take_profit = entry_price
    return stop_loss, take_profit

def calculate_risk_reward_ratio(entry_price, stop_loss, take_profit):
    if entry_price == stop_loss or entry_price == take_profit:
        return 0  # or float('inf') depending on your preference
    return abs(take_profit - entry_price) / abs(stop_loss - entry_price)

def print_summary(analysis_results):
    print(f"Analysis results for {analysis_results['symbol']}:")
    print(f"Current Price: {analysis_results['current_price']:.5f}")
    print(f"Overall Direction: {analysis_results['overall_direction']}")
    print(f"Entry Signal: {analysis_results['entry_signal']}")
    print(f"ML Prediction: {analysis_results['ml_prediction']:.2f}")
    print(f"Risk Management: {analysis_results['risk_management']}")
    
    if analysis_results['in_kill_zone']:
        print(f"Kill Zone: {analysis_results['kill_zone_info']}")
    
    print("\nKey Levels:")
    print(f"Entry Price: {analysis_results['entry_price']:.5f}")
    print(f"Stop Loss: {analysis_results['stop_loss']:.5f}")
    print(f"Take Profit: {analysis_results['take_profit']:.5f}")
    print(f"Closest Fibonacci Level: {analysis_results['closest_fib_level']:.1%}")
    print(f"Closest Fibonacci Price: {analysis_results['closest_fib_price']:.5f}")
    print(f"Trend: {analysis_results['trend']}")
    
    print("\nRecent Institutional Candles:")
    for date, direction in analysis_results['institutional_candles'][-5:]:
        print(f"{date}: {direction}")

def determine_kill_zone(current_time):
    # Define session times for different markets (all times in UTC)
    london_open = (time(8, 0), time(9, 0))
    new_york_open = (time(13, 30), time(14, 30))
    new_york_close = (time(20, 0), time(21, 0))
    asian_open = (time(0, 0), time(1, 0))  # Tokyo open

    current_time = current_time.time()
    
    if london_open[0] <= current_time <= london_open[1]:
        return "London Open Session"
    elif new_york_open[0] <= current_time <= new_york_open[1]:
        return "New York Open Session"
    elif new_york_close[0] <= current_time <= new_york_close[1]:
        return "New York Close Session"
    elif asian_open[0] <= current_time <= asian_open[1]:
        return "Asian Open Session"
    else:
        return None

def find_recent_swing_levels(data, window=10):
    highs = data['High'].rolling(window=window, center=True).max()
    lows = data['Low'].rolling(window=window, center=True).min()
    swing_highs = data['High'][(data['High'].shift(1) < data['High']) & (data['High'].shift(-1) < data['High']) & (data['High'] == highs)]
    swing_lows = data['Low'][(data['Low'].shift(1) > data['Low']) & (data['Low'].shift(-1) > data['Low']) & (data['Low'] == lows)]
    return swing_highs, swing_lows

def calculate_stop_loss(data, entry_price, trend):
    swing_highs, swing_lows = find_recent_swing_levels(data)
    
    if trend == 'bullish':
        # Find the most recent swing low below the entry price
        valid_swing_lows = swing_lows[swing_lows < entry_price]
        if not valid_swing_lows.empty:
            stop_loss = valid_swing_lows.iloc[-1]
        else:
            # If no valid swing low, use the lowest low of the last 10 periods
            stop_loss = data['Low'].tail(10).min()
    else:  # bearish
        # Find the most recent swing high above the entry price
        valid_swing_highs = swing_highs[swing_highs > entry_price]
        if not valid_swing_highs.empty:
            stop_loss = valid_swing_highs.iloc[-1]
        else:
            # If no valid swing high, use the highest high of the last 10 periods
            stop_loss = data['High'].tail(10).max()
    
    return round(stop_loss, 5)

def analyze_symbol(symbol, data, ml_model):
    try:
        print(f"Starting analysis for {symbol}...")
        
        # Ensure data is sorted by date
        data['1h'] = data['1h'].sort_index()
        data['1d'] = data['1d'].sort_index()
        
        # Calculate True Range for ATR
        data['1h']['TR'] = np.maximum(data['1h']['High'] - data['1h']['Low'], 
                                      np.maximum(abs(data['1h']['High'] - data['1h']['Close'].shift()), 
                                                 abs(data['1h']['Low'] - data['1h']['Close'].shift())))
        
        current_price = data['1h']['Close'].iloc[-1]
        current_time = data['1h'].index[-1]
        
        # Determine trend
        trend = determine_trend(data['1d'])
        
        # Calculate Fibonacci levels
        fib_levels = calculate_fibonacci_levels(data['1d'], trend)
        
        # Determine kill zone
        kill_zone_info = determine_kill_zone(current_time)
        
        # Identify order blocks and liquidity levels
        ob_highs, ob_lows = identify_order_blocks(data['1h'])
        liq_highs, liq_lows = identify_liquidity_levels(data['1h'])
        
        entry_signal = determine_entry_signal(data['1h'], trend, fib_levels, kill_zone_info, (ob_highs, ob_lows), (liq_highs, liq_lows))
        stop_loss = calculate_stop_loss(data['1h'], current_price, trend)
        take_profit = calculate_take_profit(current_price, stop_loss)
        predicted_price = predict_price_movement(data['1h'], ml_model)
        risk_assessment, position_size = assess_risk_management(stop_loss, take_profit, current_price, ACCOUNT_BALANCE)
        institutional_candles = identify_institutional_candles(data['1h'])

        analysis_results = {
            'current_price': current_price,
            'trend': trend,
            'fibonacci_levels': fib_levels,
            'kill_zone': kill_zone_info,
            'order_blocks': {'highs': ob_highs, 'lows': ob_lows},
            'liquidity_levels': {'highs': liq_highs, 'lows': liq_lows},
            'entry_signal': entry_signal,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'predicted_price': predicted_price,
            'risk_assessment': risk_assessment,
            'position_size': position_size,
            'institutional_candles': institutional_candles
        }
        
        return analysis_results
    except Exception as e:
        print(f"Error in analyze_symbol for {symbol}: {str(e)}")
        return None

def train_ml_model(all_data):
    features = []
    targets = []
    
    for symbol, data in all_data.items():
        try:
            df = data['1h']
            
            # Create ICT-based features
            df['OB_distance'] = df.apply(lambda row: calculate_ob_distance(df.loc[:row.name]), axis=1)
            df['Liq_level_distance'] = df.apply(lambda row: calculate_liq_level_distance(df.loc[:row.name]), axis=1)
            df['Trend'] = determine_trend(data['1d'])
            df['Kill_zone'] = df.index.map(lambda x: int(determine_kill_zone(x)['in_kill_zone']))
            df['Institutional_candle'] = df.apply(lambda row: int(bool(identify_institutional_candles(df.loc[:row.name]))), axis=1)
            
            feature_cols = ['OB_distance', 'Liq_level_distance', 'Trend', 'Kill_zone', 'Institutional_candle']
            symbol_features = df[feature_cols]
            
            # Check for infinite or NaN values
            if symbol_features.isnull().values.any() or np.isinf(symbol_features.values).any():
                print(f"Skipping {symbol} due to invalid values")
                continue
            
            features.append(symbol_features)
            
            # Create target based on future price movement
            df['target'] = (df['Close'].shift(-10) > df['Close']).astype(int)
            targets.append(df['target'])
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    if not features or not targets:
        raise ValueError("No valid data to train the model")
    
    X = pd.concat(features, axis=0)
    y = pd.concat(targets, axis=0)
    
    # Remove any rows with NaN values
    valid_data = X.dropna().index
    X = X.loc[valid_data]
    y = y.loc[valid_data]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def main():
    ml_confidence_threshold = 0.7  # Set a default threshold, or adjust as needed
    
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    intervals = ['1h', '1d', '1wk', '1mo']
    
    all_data = {}
    for symbol in symbols:
        print(f"\nFetching data for {symbol}...")
        symbol_data = fetch_and_process_data(symbol, start_date, end_date, intervals)
        if symbol_data:
            all_data[symbol] = symbol_data

    print("\nTraining ML model...")
    ml_model = train_ml_model(all_data)

    for symbol, data in all_data.items():
        if all(len(df) > 0 for df in data.values()):
            analysis_results = analyze_symbol(symbol, data, ml_model)
            if analysis_results:
                print(f"\nAnalysis Results for {symbol}:")
                print(f"Current Price: {analysis_results['current_price']:.5f}")
                print(f"Overall Direction: {analysis_results['overall_direction']}")
                print(f"Entry Signal: {analysis_results['entry_signal']}")
                print(f"ML Prediction: {analysis_results['ml_prediction']:.2f}")
                print(f"Risk Management: {analysis_results['risk_management']}")
                print("\nKey Levels:")
                print(f"Entry Price: {analysis_results['key_levels']['entry']:.5f}")
                print(f"Stop Loss: {analysis_results['key_levels']['stop_loss']:.5f}")
                print(f"Take Profit: {analysis_results['key_levels']['take_profit']:.5f}")
                print(f"Closest Fibonacci Level: {analysis_results['key_levels']['closest_fib_level']:.3f}")
                print(f"Closest Fibonacci Price: {analysis_results['key_levels']['closest_fib_price']:.5f}")
                print("\nKill Zone Info:")
                print(f"In Kill Zone: {analysis_results['kill_zone_info']['in_kill_zone']}")
                print(f"Session: {analysis_results['kill_zone_info']['session']}")
                print("\nRecent Institutional Candles:")
                for candle in analysis_results['recent_institutional_candles']:
                    print(f"Type: {candle['type']}, Time: {candle['time']}")
                print("\nOrder Blocks:")
                for ob in analysis_results['order_blocks']:
                    print(f"Type: {ob['type']}, Price: {ob['price']:.5f}, Time: {ob['time']}")
                print("\nLiquidity Levels:")
                for ll in analysis_results['liquidity_levels']:
                    print(f"Type: {ll['type']}, Price: {ll['price']:.5f}, Time: {ll['time']}")
                print("\nFibonacci Levels:")
                for level, price in analysis_results['fib_levels'].items():
                    print(f"{level}: {price:.5f}")
            else:
                print(f"Analysis failed for {symbol}")
        else:
            print(f"Insufficient data for {symbol}")

    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()