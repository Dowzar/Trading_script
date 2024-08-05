import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def fetch_data(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        if df.empty:
            print(f"No data available for {symbol} at interval {interval}")
            return None
        df['Symbol'] = symbol
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol} at interval {interval}: {str(e)}")
        return None

def determine_overall_direction(data: Dict[str, pd.DataFrame]) -> str:
    # Sort intervals from largest to smallest
    sorted_intervals = sorted(data.keys(), key=lambda x: {
        '1mo': 0, '1wk': 1, '1d': 2, '1h': 3, '15m': 4
    }.get(x, 5))

    # Use the two largest timeframes for top-down analysis
    top_timeframes = sorted_intervals[:2]

    direction_scores = []

    for interval in top_timeframes:
        df = data[interval]
        
        # ICT Concept: Fair Value Gap (FVG)
        df['FVG_bullish'] = (df['Low'].shift(2) > df['High'].shift(1)) & (df['Open'] < df['Low'].shift(2))
        df['FVG_bearish'] = (df['High'].shift(2) < df['Low'].shift(1)) & (df['Open'] > df['High'].shift(2))
        
        # ICT Concept: Optimal Trade Entry (OTE)
        df['OTE_bullish'] = (df['Low'] <= df['EMA_200']) & (df['Close'] > df['EMA_200'])
        df['OTE_bearish'] = (df['High'] >= df['EMA_200']) & (df['Close'] < df['EMA_200'])
        
        # ICT Concept: Breaker Blocks
        df['breaker_bullish'] = (df['High'] > df['High'].rolling(10).max().shift()) & (df['Close'] < df['Open'])
        df['breaker_bearish'] = (df['Low'] < df['Low'].rolling(10).min().shift()) & (df['Close'] > df['Open'])
        
        # ICT Concept: Institutional Candles
        df['inst_bullish'] = (df['Close'] > df['Open']) & ((df['High'] - df['Close']) / (df['High'] - df['Low']) < 0.3)
        df['inst_bearish'] = (df['Close'] < df['Open']) & ((df['Close'] - df['Low']) / (df['High'] - df['Low']) < 0.3)

        # ICT Concept: Order Blocks
        df['order_block_bullish'] = (df['Low'] < df['Low'].shift(1)) & (df['High'] > df['High'].shift(1)) & (df['Close'] > df['Open'])
        df['order_block_bearish'] = (df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1)) & (df['Close'] < df['Open'])

        # Calculate direction score
        bullish_score = sum([
            df['FVG_bullish'].iloc[-1],
            df['OTE_bullish'].iloc[-1],
            df['breaker_bullish'].iloc[-1],
            df['inst_bullish'].iloc[-1],
            df['order_block_bullish'].iloc[-1],
            df['Close'].iloc[-1] > df['EMA_200'].iloc[-1],
            df['Close'].iloc[-1] > df['EMA_50'].iloc[-1],
            df['EMA_50'].iloc[-1] > df['EMA_200'].iloc[-1]
        ])
        
        bearish_score = sum([
            df['FVG_bearish'].iloc[-1],
            df['OTE_bearish'].iloc[-1],
            df['breaker_bearish'].iloc[-1],
            df['inst_bearish'].iloc[-1],
            df['order_block_bearish'].iloc[-1],
            df['Close'].iloc[-1] < df['EMA_200'].iloc[-1],
            df['Close'].iloc[-1] < df['EMA_50'].iloc[-1],
            df['EMA_50'].iloc[-1] < df['EMA_200'].iloc[-1]
        ])
        
        # Weight the score based on the timeframe (larger timeframe has more weight)
        timeframe_weight = len(top_timeframes) - top_timeframes.index(interval)
        direction_scores.append((bullish_score - bearish_score) * timeframe_weight)

    # Determine overall direction based on weighted scores
    overall_score = sum(direction_scores)
    
    if overall_score > 1:
        return "Bullish"
    elif overall_score < -1:
        return "Bearish"
    else:
        return "Neutral"

def check_entry_conditions(data, overall_direction, ml_prediction, ml_confidence_threshold):
    ict_signal = "HOLD NO BUY or SELL SIGNAL YET"
    position = "Neutral"
    
    if overall_direction == "Bullish":
        if (data['Close'] > data['EMA_21'] and
            data['OTE_Long'] and
            data['FVG_up'] and
            data['Institutional_Candle'] and
            data['Liquidity_Sweep_Low'] and
            data['Breaker_Bullish'] and
            data['Low'] < data['Mitigation_Block_Low'] and
            (data['Breaker_Block_Bullish'] or data['Rejection_Block_Bullish'] or 
             data['Divergence_Phantom_Bullish'] or data['Double_Bottom'])):
            ict_signal = "BUY"
            position = "Long"
    elif overall_direction == "Bearish":
        if (data['Close'] < data['EMA_21'] and
            data['OTE_Short'] and
            data['FVG_down'] and
            data['Institutional_Candle'] and
            data['Liquidity_Sweep_High'] and
            data['Breaker_Bearish'] and
            data['High'] > data['Mitigation_Block_High'] and
            (data['Breaker_Block_Bearish'] or data['Rejection_Block_Bearish'] or 
             data['Divergence_Phantom_Bearish'] or data['Double_Top'])):
            ict_signal = "SELL"
            position = "Short"
    
    # Incorporate ML prediction
    if ict_signal == "BUY" and ml_prediction < ml_confidence_threshold:
        return "HOLD (ICT: BUY, ML: Uncertain)", "Neutral"
    elif ict_signal == "SELL" and ml_prediction > (1 - ml_confidence_threshold):
        return "HOLD (ICT: SELL, ML: Uncertain)", "Neutral"
    elif ict_signal == "HOLD NO BUY or SELL SIGNAL YET" and ml_prediction > ml_confidence_threshold:
        return "HOLD (ICT: No Signal, ML: Bullish)", "Neutral"
    elif ict_signal == "HOLD NO BUY or SELL SIGNAL YET" and ml_prediction < (1 - ml_confidence_threshold):
        return "HOLD (ICT: No Signal, ML: Bearish)", "Neutral"
    
    return ict_signal, position

def calculate_entry_stop_target(m15, signal):
    entry = m15['Close']
    
    if signal == "BUY":
        stop = m15['Low'] - (m15['High'] - m15['Low']) * 0.1
        target = entry + (entry - stop) * 2
    else:  # SELL
        stop = m15['High'] + (m15['High'] - m15['Low']) * 0.1
        target = entry - (stop - entry) * 2
    
    return entry, stop, target

def find_closest_fib(m15, m15_data):
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    recent_high = m15_data['High'].rolling(window=100).max().iloc[-1]
    recent_low = m15_data['Low'].rolling(window=100).min().iloc[-1]
    
    current_price = m15['Close']
    fib_prices = [recent_low + level * (recent_high - recent_low) for level in fib_levels]
    
    closest_fib = min(fib_levels, key=lambda x: abs(recent_low + x * (recent_high - recent_low) - current_price))
    return f"Fib_{closest_fib}"

def calculate_risk_management(entry, stop, target, balance=10000, risk_percent=1):
    if entry is None or stop is None or target is None:
        return {}
    
    risk_amount = balance * (risk_percent / 100)
    position_size = risk_amount / abs(entry - stop)
    potential_profit = position_size * abs(target - entry)
    risk_reward_ratio = abs(target - entry) / abs(stop - entry)
    
    return {
        'Position Size': position_size,
        'Potential Profit': potential_profit,
        'Risk-Reward Ratio': risk_reward_ratio,
        'Risk Amount': risk_amount
    }

def get_available_intervals(symbol: str) -> List[str]:
    all_intervals = ['1mo', '1wk', '1d', '1h', '15m']
    available_intervals = []
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    for interval in all_intervals:
        if interval in ['15m', '1h']:
            start_date = end_date - timedelta(days=60)  # 60 days of data for shorter intervals
        df = fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), interval)
        if df is not None and not df.empty:
            available_intervals.append(interval)
    
    return available_intervals

def calculate_ict_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate EMA 21
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # Calculate OTE (Optimal Trade Entry)
    df['OTE_Long'] = (df['Low'] < df['Low'].shift(1)) & (df['Close'] > df['Close'].shift(1))
    df['OTE_Short'] = (df['High'] > df['High'].shift(1)) & (df['Close'] < df['Close'].shift(1))
    
    # Calculate FVG (Fair Value Gap)
    df['FVG_up'] = (df['Low'] > df['High'].shift(1))
    df['FVG_down'] = (df['High'] < df['Low'].shift(1))
    
    # Institutional Candle (assuming large body candles)
    df['Candle_Size'] = abs(df['Close'] - df['Open'])
    df['Institutional_Candle'] = df['Candle_Size'] > df['Candle_Size'].rolling(window=20).mean() * 1.5
    
    # Liquidity Sweep
    df['Liquidity_Sweep_Low'] = (df['Low'] < df['Low'].rolling(window=20).min().shift(1)) & (df['Close'] > df['Open'])
    df['Liquidity_Sweep_High'] = (df['High'] > df['High'].rolling(window=20).max().shift(1)) & (df['Close'] < df['Open'])
    
    # Breaker
    df['Breaker_Bullish'] = (df['High'] > df['High'].shift(1)) & (df['Low'].shift(-1) > df['High']) & (df['Close'] > df['Open'])
    df['Breaker_Bearish'] = (df['Low'] < df['Low'].shift(1)) & (df['High'].shift(-1) < df['Low']) & (df['Close'] < df['Open'])
    
    # Mitigation Block
    df['Mitigation_Block_High'] = df['High'].rolling(window=10).max()
    df['Mitigation_Block_Low'] = df['Low'].rolling(window=10).min()
    
    # Handle timezone conversion
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC').tz_convert('Africa/Johannesburg')
    else:
        df.index = df.index.tz_convert('Africa/Johannesburg')

    df['Hour'] = df.index.hour
    df['KillZone_London_Open'] = (df['Hour'] >= 9) & (df['Hour'] < 11)  # 09:00-11:00 SAST
    df['KillZone_NewYork_Open'] = (df['Hour'] >= 15) & (df['Hour'] < 17)  # 15:00-17:00 SAST
    df['KillZone_Asian_Open'] = (df['Hour'] >= 2) & (df['Hour'] < 4)  # 02:00-04:00 SAST
    df['KillZone_NewYork_Close'] = (df['Hour'] >= 22) | (df['Hour'] < 0)  # 22:00-00:00 SAST
    df['InKillZone'] = (df['KillZone_London_Open'] | df['KillZone_NewYork_Open'] | 
                        df['KillZone_Asian_Open'] | df['KillZone_NewYork_Close'])
    
    # Breaker Blocks
    df['Breaker_Block_High'] = df['High'].rolling(window=5).max()
    df['Breaker_Block_Low'] = df['Low'].rolling(window=5).min()
    df['Breaker_Block_Bullish'] = (df['Close'] > df['Breaker_Block_High'].shift(1)) & (df['Open'] < df['Breaker_Block_High'].shift(1))
    df['Breaker_Block_Bearish'] = (df['Close'] < df['Breaker_Block_Low'].shift(1)) & (df['Open'] > df['Breaker_Block_Low'].shift(1))

    # ICT Rejection Block
    df['Rejection_Block_High'] = df['High'].rolling(window=3).max()
    df['Rejection_Block_Low'] = df['Low'].rolling(window=3).min()
    df['Rejection_Block_Bullish'] = (df['Low'] < df['Rejection_Block_Low'].shift(1)) & (df['Close'] > df['Open'])
    df['Rejection_Block_Bearish'] = (df['High'] > df['Rejection_Block_High'].shift(1)) & (df['Close'] < df['Open'])

    # ICT Divergence Phantoms
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    df['Price_Higher_High'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
    df['Price_Lower_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
    df['RSI_Lower_High'] = (df['RSI'] < df['RSI'].shift(1)) & (df['RSI'].shift(1) < df['RSI'].shift(2))
    df['RSI_Higher_Low'] = (df['RSI'] > df['RSI'].shift(1)) & (df['RSI'].shift(1) > df['RSI'].shift(2))
    df['Divergence_Phantom_Bullish'] = df['Price_Lower_Low'] & df['RSI_Higher_Low']
    df['Divergence_Phantom_Bearish'] = df['Price_Higher_High'] & df['RSI_Lower_High']

    # ICT Double Bottom and Tops
    df['Double_Bottom'] = (df['Low'] == df['Low'].rolling(window=10).min()) & (df['Low'].shift(5) == df['Low'].rolling(window=10).min())
    df['Double_Top'] = (df['High'] == df['High'].rolling(window=10).max()) & (df['High'].shift(5) == df['High'].rolling(window=10).max())

    return df

def calculate_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_and_process_data(symbol: str, start_date: str, end_date: str, intervals: List[str]) -> Dict[str, pd.DataFrame]:
    data = {}
    for interval in intervals:
        try:
            df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            if not df.empty:
                df = calculate_ict_indicators(df)
                data[interval] = df
            else:
                print(f"No data available for {symbol} at interval {interval}")
        except Exception as e:
            print(f"Error fetching data for {symbol} at interval {interval}: {str(e)}")
    
    if not data:
        print(f"No data available for {symbol} at any interval")
        return None
    
    available_intervals = list(data.keys())
    print(f"Available intervals for {symbol}: {available_intervals}")
    return data

def prepare_data_for_ml(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Prepare features
    features = [
        'EMA_21', 'OTE_Long', 'OTE_Short', 'FVG_up', 'FVG_down',
        'Institutional_Candle', 'Liquidity_Sweep_Low', 'Liquidity_Sweep_High',
        'Breaker_Bullish', 'Breaker_Bearish', 'Breaker_Block_Bullish',
        'Breaker_Block_Bearish', 'Rejection_Block_Bullish', 'Rejection_Block_Bearish',
        'Divergence_Phantom_Bullish', 'Divergence_Phantom_Bearish',
        'Double_Bottom', 'Double_Top', 'RSI'
    ]
    X = df[features]
    
    # Prepare target
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # 1 for price increase, 0 for decrease
    y = df['Target']
    
    return X, y

def train_ml_model(data: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[RandomForestClassifier, StandardScaler]:
    all_X = pd.DataFrame()
    all_y = pd.Series()
    
    for symbol, symbol_data in data.items():
        for interval, df in symbol_data.items():
            X, y = prepare_data_for_ml(df)
            all_X = pd.concat([all_X, X])
            all_y = pd.concat([all_y, y])
    
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    accuracy = model.score(X_test_scaled, y_test)
    print(f"ML Model Accuracy: {accuracy:.2f}")
    
    return model, scaler

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

    available_intervals = sorted(data.keys(), key=lambda x: {
        '1mo': 0, '1wk': 1, '1d': 2, '1h': 3, '15m': 4
    }.get(x, 5), reverse=True)

    # Use the shortest available timeframe for entry signals
    entry_interval = available_intervals[0]
    df = data[entry_interval]
    current_price = df['Close'].iloc[-1]
    
    # Determine overall direction using ICT Concepts and top-down analysis
    overall_direction = determine_overall_direction(data)
    
    # Entry signal logic (using the shortest available timeframe)
    entry_signal = "HOLD"
    if overall_direction == "Bullish":
        if df['Close'].iloc[-1] > df['EMA_50'].iloc[-1] and df['Close'].iloc[-2] <= df['EMA_50'].iloc[-2]:
            entry_signal = "BUY"
    elif overall_direction == "Bearish":
        if df['Close'].iloc[-1] < df['EMA_50'].iloc[-1] and df['Close'].iloc[-2] >= df['EMA_50'].iloc[-2]:
            entry_signal = "SELL"

    # ML prediction
    features = extract_features(df)
    ml_prediction = ml_model.predict_proba(scaler.transform(features.reshape(1, -1)))[0][1]

    # Combine ICT and ML signals
    if ml_prediction > ml_confidence_threshold:
        if entry_signal == "BUY":
            entry_signal = "STRONG BUY"
        elif entry_signal == "HOLD" and overall_direction == "Bullish":
            entry_signal = "BUY"
    elif ml_prediction < (1 - ml_confidence_threshold):
        if entry_signal == "SELL":
            entry_signal = "STRONG SELL"
        elif entry_signal == "HOLD" and overall_direction == "Bearish":
            entry_signal = "SELL"

    # ... (rest of the function remains the same)

    # ... (rest of the function remains the same)

def main():
    symbols = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
        "USDCHF=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
        "EURCHF=X", "AUDJPY=X", "CADJPY=X", "NZDJPY=X", "GBPCAD=X",
        "GBPCHF=X", "EURAUD=X", "EURCAD=X", "AUDCAD=X", "AUDCHF=X",
        "AUDNZD=X", "NZDCAD=X", "NZDCHF=X", "CADCHF=X", "EURNZD=X",
        "GBPAUD=X", "GBPNZD=X", "USDMXN=X", "USDZAR=X", "USDTRY=X",
        "USDSEK=X", "USDNOK=X", "USDDKK=X", "USDSGD=X", "USDHKD=X",
        "USDRUB=X", "USDCNH=X", "USDPLN=X", "USDINR=X", "USDBRL=X",
        "USDCZK=X", "USDHUF=X", "USDILS=X", "GC=F", "SI=F", "PA=F",
        "PL=F", "CL=F", "NG=F", "RB=F", "HO=F", "BZ=F", "^GSPC",
        "^DJI", "^IXIC", "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI",
        "^BSESN", "^AXJO"
    ]
    
    intervals = ['15m', '1h', '1d', '1wk', '1mo']  # Order from shortest to longest
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    ml_confidence_threshold = float(input("Enter ML confidence threshold (0.5 to 1.0): "))
    
    all_data = {}
    for symbol in symbols:
        symbol_data = fetch_and_process_data(symbol, start_date, end_date, intervals)
        if symbol_data:
            all_data[symbol] = symbol_data
    
    ml_model, scaler = train_ml_model(all_data)
    
    for symbol, data in all_data.items():
        analysis_results = analyze_data(data, ml_model, scaler, ml_confidence_threshold)
        if analysis_results:
            print(f"\nAnalysis results for {symbol}:")
            print(f"Current Price: {analysis_results['current_price']:.5f}")
            print(f"Overall direction: {analysis_results['overall_direction']}")
            print(f"Entry Signal: {analysis_results['entry_signal']}")
            print(f"Position: {analysis_results['position']}")
            print(f"ML Prediction (Probability of price increase): {analysis_results['ml_prediction']:.2f}")
            print(f"ML Confidence Threshold: {ml_confidence_threshold:.2f}")
            
            if "BUY" in analysis_results['entry_signal'] or "SELL" in analysis_results['entry_signal']:
                print(f"Entry Price: {analysis_results['entry']:.5f}")
                print(f"Stop: {analysis_results['stop']:.5f}")
                print(f"Target: {analysis_results['target']:.5f}")
                print(f"Closest Fibonacci Level: {analysis_results['closest_fib']}")
                print(f"Risk Management: {analysis_results['risk_management']}")
            
            print(f"Additional ICT Concepts:")
            print(f"Breaker Blocks: {analysis_results['breaker_blocks']}")
            print(f"Rejection Blocks: {analysis_results['rejection_blocks']}")
            print(f"Divergence Phantoms: {analysis_results['divergence_phantoms']}")
            print(f"Double Patterns: {analysis_results['double_patterns']}")
            print(f"Kill Zone Info: {analysis_results['kill_zone_info']}")
            print(f"Currently in Kill Zone: {'Yes' if analysis_results['in_kill_zone'] else 'No'}")
        else:
            print(f"Skipping analysis for {symbol} due to data fetching issues.")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()