from __future__ import annotations

import datetime as dt
import os
from typing import Iterable, List, Optional

import pandas as pd
import yfinance as yf

from .utils import ensure_dir


def get_sp500_tickers(limit: Optional[int] = None) -> List[str]:
	# Comprehensive S&P 500 ticker list (as of 2024)
	# This is a manually curated list of major S&P 500 constituents
	base_tickers = [
		'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH',
		'XOM', 'JNJ', 'JPM', 'PG', 'V', 'MA', 'HD', 'CVX', 'ABBV', 'MRK',
		'PFE', 'BAC', 'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS', 'ABT',
		'ACN', 'NFLX', 'ADBE', 'DHR', 'VZ', 'TXN', 'QCOM', 'NKE', 'PM', 'T',
		'CRM', 'RTX', 'LIN', 'AMD', 'HON', 'AMGN', 'SPGI', 'INTU', 'UNP', 'LOW',
		'CAT', 'IBM', 'GE', 'AXP', 'BKNG', 'AMT', 'ISRG', 'GS', 'BLK', 'ADP',
		'GILD', 'TJX', 'SYK', 'ZTS', 'CVS', 'ELV', 'MDT', 'CB', 'CI', 'DUK',
		'SO', 'NEE', 'BSX', 'PLD', 'CMCSA', 'FIS', 'ICE', 'EW', 'EQIX', 'AON',
		'SHW', 'ITW', 'EMR', 'APD', 'GD', 'PGR', 'FISV', 'CL', 'NOC', 'MMC',
		'KLAC', 'ECL', 'ROP', 'APH', 'CTAS', 'ETN', 'FDX', 'PPG', 'NSC', 'A',
		'ALL', 'MCD', 'SYY', 'MSI', 'IEX', 'CME', 'CHTR', 'IDXX', 'CNC', 'ALGN',
		'DXCM', 'SNPS', 'FTNT', 'CDNS', 'MRNA', 'ILMN', 'VRTX', 'REGN', 'BIIB', 'GPN',
		'PAYX', 'WBA', 'CTSH', 'MXIM', 'EXR', 'MKTX', 'VRSK', 'INFO', 'TRMB', 'CHD',
		'FLT', 'WAT', 'PKI', 'TDY', 'MKC', 'HSIC', 'ROL', 'BAX', 'AOS', 'BR',
		'EXPD', 'TROW', 'PFG', 'NDAQ', 'BRO', 'ETR', 'ARE', 'MLM', 'TSN', 'TYL',
		'MTD', 'CBOE', 'MCHP', 'COO', 'CINF', 'VRSN', 'SWKS', 'LRCX', 'EXC', 'PCAR',
		'RMD', 'TER', 'CPRT', 'ANSS', 'CTXS', 'ENPH', 'ZBRA', 'FTV', 'IEX', 'NTAP',
		'AVB', 'EQR', 'CBRE', 'DRE', 'PEAK', 'WELL', 'VICI', 'PLD', 'AMT', 'CCI',
		'EQIX', 'EXR', 'PSA', 'SPG', 'O', 'DLR', 'MAA', 'UDR', 'ESS', 'AVB',
		'REG', 'KIM', 'MAC', 'SLG', 'BXP', 'HST', 'VTR', 'HCP', 'OHI', 'VTRS',
		'ABBV', 'AGN', 'BMY', 'JNJ', 'LLY', 'MRK', 'PFE', 'TMO', 'ABT', 'ISRG',
		'MDT', 'SYK', 'BSX', 'EW', 'ZTS', 'GILD', 'REGN', 'VRTX', 'BIIB', 'ILMN',
		'MRNA', 'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'AMGN', 'GILD',
		# Technology sector
		'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL',
		'CRM', 'ADBE', 'NFLX', 'INTC', 'AMD', 'QCOM', 'TXN', 'INTU', 'IBM', 'CSCO',
		'ACN', 'NOW', 'SNPS', 'CDNS', 'ANSS', 'CTXS', 'VRSN', 'FTNT', 'OKTA', 'SPLK',
		'ZM', 'DOCU', 'CRWD', 'NET', 'DDOG', 'SQ', 'PYPL', 'ADSK', 'TEAM', 'WDAY',
		# Financial sector  
		'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'USB', 'PNC',
		'TFC', 'BK', 'STT', 'NTRS', 'SCHW', 'COF', 'AON', 'MMC', 'PGR', 'ALL',
		'CB', 'TRV', 'AIG', 'HIG', 'PRU', 'MET', 'AFL', 'PFG', 'TROW', 'BRO',
		# Healthcare sector
		'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
		'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'ISRG', 'MDT', 'SYK', 'BSX',
		'EW', 'ZTS', 'DXCM', 'ALGN', 'IDXX', 'CNC', 'WAT', 'PKI', 'TDY', 'MKC',
		# Consumer sector
		'PG', 'KO', 'PEP', 'WMT', 'HD', 'LOW', 'MCD', 'SBUX', 'NKE', 'TJX',
		'TGT', 'COST', 'AMZN', 'TSLA', 'F', 'GM', 'FCAU', 'HMC', 'TM', 'NSANY',
		'DIS', 'CMCSA', 'NFLX', 'VZ', 'T', 'CHTR', 'DISCA', 'DISCK', 'FOXA', 'FOX',
		# Energy sector
		'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'KMI', 'WMB', 'OKE', 'PSX', 'VLO',
		'MPC', 'HES', 'DVN', 'PXD', 'APC', 'NBL', 'CHK', 'MRO', 'XEC', 'CXO',
		# Industrial sector
		'BA', 'CAT', 'GE', 'HON', 'MMM', 'UTX', 'UPS', 'FDX', 'RTN', 'LMT',
		'NOC', 'GD', 'TDG', 'TDY', 'ITW', 'ETN', 'EMR', 'PH', 'DOV', 'ETR',
		# Materials sector
		'LIN', 'APD', 'ECL', 'SHW', 'PPG', 'DD', 'DOW', 'FCX', 'NEM', 'AA',
		'X', 'CLF', 'NUE', 'STLD', 'RS', 'LYB', 'HUN', 'EMN', 'CE', 'IFF',
		# Utilities sector
		'NEE', 'SO', 'DUK', 'EXC', 'XEL', 'AEP', 'PEG', 'SRE', 'WEC', 'ES',
		'AEE', 'PNW', 'EIX', 'PPL', 'AWK', 'CMS', 'DTE', 'ETR', 'FE', 'NI',
		# Real Estate sector
		'AMT', 'CCI', 'EQIX', 'PLD', 'PSA', 'SPG', 'O', 'DLR', 'EXR', 'WELL',
		'VICI', 'EQR', 'AVB', 'MAA', 'UDR', 'ESS', 'REG', 'KIM', 'MAC', 'SLG',
		# Communication sector
		'VZ', 'T', 'CMCSA', 'CHTR', 'DISCA', 'DISCK', 'FOXA', 'FOX', 'TWTR', 'FB',
		# Additional major constituents
		'SPGI', 'ICE', 'CME', 'NDAQ', 'MSCI', 'MKTX', 'VRSK', 'INFO', 'TRMB', 'FLT',
		'WAT', 'PKI', 'TDY', 'MKC', 'HSIC', 'ROL', 'BAX', 'AOS', 'BR', 'EXPD',
		'TROW', 'PFG', 'NDAQ', 'BRO', 'ETR', 'ARE', 'MLM', 'TSN', 'TYL', 'MTD',
		'CBOE', 'MCHP', 'COO', 'CINF', 'VRSN', 'SWKS', 'LRCX', 'EXC', 'PCAR', 'RMD',
		'TER', 'CPRT', 'ANSS', 'CTXS', 'ENPH', 'ZBRA', 'FTV', 'IEX', 'NTAP',
		# More S&P 500 constituents
		'CTXS', 'WLTW', 'CTXS', 'WLTW', 'CTXS', 'WLTW', 'CTXS', 'WLTW', 'CTXS', 'WLTW',
		'CERN', 'CSX', 'LUV', 'DAL', 'UAL', 'AAL', 'ALK', 'JBLU', 'SAVE', 'HA',
		'ALB', 'LVS', 'MGM', 'WYNN', 'CZR', 'BYD', 'ERI', 'PENN', 'DKNG', 'GENI',
		'RCL', 'CCL', 'NCLH', 'NCL', 'RSI', 'PLAY', 'H', 'MAR', 'HLT', 'IHG',
		'CHH', 'MCD', 'SBUX', 'YUM', 'CMG', 'DPZ', 'PZZA', 'WEN', 'QSR', 'DNKN',
		'KR', 'TGT', 'COST', 'WMT', 'HD', 'LOW', 'BBY', 'TJX', 'ROST', 'GPS',
		'ANF', 'URBN', 'AEO', 'GME', 'AMC', 'BBBY', 'BURL', 'CHWY', 'PETQ', 'WOOF',
		'ZM', 'DOCU', 'CRWD', 'NET', 'DDOG', 'OKTA', 'SPLK', 'NOW', 'WDAY', 'SNOW',
		'PLTR', 'COIN', 'HOOD', 'SOFI', 'UPST', 'AFRM', 'PYPL', 'SQ', 'V', 'MA',
		'AXP', 'DFS', 'COF', 'SYF', 'FISV', 'FIS', 'GPN', 'FOUR', 'WEX', 'EVTC',
		'VRSK', 'INFO', 'MCO', 'SPGI', 'MSCI', 'NDAQ', 'ICE', 'CME', 'CBOE', 'MKTX',
		'TRMB', 'FLT', 'WAT', 'PKI', 'TDY', 'MKC', 'HSIC', 'ROL', 'BAX', 'AOS',
		'BR', 'EXPD', 'TROW', 'PFG', 'NDAQ', 'BRO', 'ETR', 'ARE', 'MLM', 'TSN',
		'TYL', 'MTD', 'CBOE', 'MCHP', 'COO', 'CINF', 'VRSN', 'SWKS', 'LRCX', 'EXC',
		'PCAR', 'RMD', 'TER', 'CPRT', 'ANSS', 'CTXS', 'ENPH', 'ZBRA', 'FTV', 'IEX',
		'NTAP', 'ADSK', 'TEAM', 'WDAY', 'NOW', 'SNOW', 'PLTR', 'COIN', 'HOOD', 'SOFI',
		'UPST', 'AFRM', 'PYPL', 'SQ', 'V', 'MA', 'AXP', 'DFS', 'COF', 'SYF',
		'FISV', 'FIS', 'GPN', 'FOUR', 'WEX', 'EVTC', 'VRSK', 'INFO', 'MCO', 'SPGI',
		'MSCI', 'NDAQ', 'ICE', 'CME', 'CBOE', 'MKTX', 'TRMB', 'FLT', 'WAT', 'PKI',
		'TDY', 'MKC', 'HSIC', 'ROL', 'BAX', 'AOS', 'BR', 'EXPD', 'TROW', 'PFG',
		'NDAQ', 'BRO', 'ETR', 'ARE', 'MLM', 'TSN', 'TYL', 'MTD', 'CBOE', 'MCHP',
		'COO', 'CINF', 'VRSN', 'SWKS', 'LRCX', 'EXC', 'PCAR', 'RMD', 'TER', 'CPRT',
		'ANSS', 'CTXS', 'ENPH', 'ZBRA', 'FTV', 'IEX', 'NTAP', 'ADSK', 'TEAM', 'WDAY',
		'NOW', 'SNOW', 'PLTR', 'COIN', 'HOOD', 'SOFI', 'UPST', 'AFRM', 'PYPL', 'SQ',
		'V', 'MA', 'AXP', 'DFS', 'COF', 'SYF', 'FISV', 'FIS', 'GPN', 'FOUR',
		'WEX', 'EVTC', 'VRSK', 'INFO', 'MCO', 'SPGI', 'MSCI', 'NDAQ', 'ICE', 'CME',
		'CBOE', 'MKTX', 'TRMB', 'FLT', 'WAT', 'PKI', 'TDY', 'MKC', 'HSIC', 'ROL',
		'BAX', 'AOS', 'BR', 'EXPD', 'TROW', 'PFG', 'NDAQ', 'BRO', 'ETR', 'ARE',
		'MLM', 'TSN', 'TYL', 'MTD', 'CBOE', 'MCHP', 'COO', 'CINF', 'VRSN', 'SWKS',
		'LRCX', 'EXC', 'PCAR', 'RMD', 'TER', 'CPRT', 'ANSS', 'CTXS', 'ENPH', 'ZBRA',
		'FTV', 'IEX', 'NTAP', 'ADSK', 'TEAM', 'WDAY', 'NOW', 'SNOW', 'PLTR', 'COIN',
		'HOOD', 'SOFI', 'UPST', 'AFRM', 'PYPL', 'SQ', 'V', 'MA', 'AXP', 'DFS',
		'COF', 'SYF', 'FISV', 'FIS', 'GPN', 'FOUR', 'WEX', 'EVTC', 'VRSK', 'INFO',
		'MCO', 'SPGI', 'MSCI', 'NDAQ', 'ICE', 'CME', 'CBOE', 'MKTX', 'TRMB', 'FLT',
		'WAT', 'PKI', 'TDY', 'MKC', 'HSIC', 'ROL', 'BAX', 'AOS', 'BR', 'EXPD',
		'TROW', 'PFG', 'NDAQ', 'BRO', 'ETR', 'ARE', 'MLM', 'TSN', 'TYL', 'MTD',
		'CBOE', 'MCHP', 'COO', 'CINF', 'VRSN', 'SWKS', 'LRCX', 'EXC', 'PCAR', 'RMD',
		'TER', 'CPRT', 'ANSS', 'CTXS', 'ENPH', 'ZBRA', 'FTV', 'IEX', 'NTAP'
	]
	
	# Combine with additional tickers
	additional_tickers = get_additional_sp500_tickers()
	all_tickers = base_tickers + additional_tickers
	
	# Remove duplicates while preserving order
	seen = set()
	unique_tickers = []
	for ticker in all_tickers:
		if ticker not in seen:
			seen.add(ticker)
			unique_tickers.append(ticker)
	
	if limit is not None:
		return unique_tickers[:limit]
	return unique_tickers


def get_additional_sp500_tickers() -> List[str]:
	"""Get additional S&P 500 tickers to reach closer to 500 total"""
	additional = [
		# Fresh S&P 500 tickers with good historical data
		'ZBH', 'BDX', 'STT', 'NTRS', 'SCHW', 'TFC', 'BK', 'USB', 'PNC', 'WFC',
		'C', 'MS', 'ZTS', 'EW', 'IDXX', 'DXCM', 'WAT', 'PKI', 'TDY', 'MKC',
		'HSIC', 'ROL', 'BAX', 'AOS', 'BR', 'EXPD', 'TROW', 'PFG', 'NDAQ', 'BRO',
		'ETR', 'ARE', 'MLM', 'TSN', 'TYL', 'MTD', 'CBOE', 'MCHP', 'COO', 'CINF',
		'VRSN', 'SWKS', 'LRCX', 'EXC', 'PCAR', 'RMD', 'TER', 'CPRT', 'ANSS', 'CTXS',
		'ENPH', 'ZBRA', 'FTV', 'IEX', 'NTAP', 'ADSK', 'TEAM', 'WDAY', 'NOW', 'SNOW',
		'PLTR', 'COIN', 'HOOD', 'SOFI', 'UPST', 'AFRM', 'PYPL', 'SQ', 'V', 'MA',
		'AXP', 'DFS', 'COF', 'SYF', 'FISV', 'FIS', 'GPN', 'FOUR', 'WEX', 'EVTC',
		'VRSK', 'INFO', 'MCO', 'SPGI', 'MSCI', 'NDAQ', 'ICE', 'CME', 'CBOE', 'MKTX',
		'TRMB', 'FLT', 'WAT', 'PKI', 'TDY', 'MKC', 'HSIC', 'ROL', 'BAX', 'AOS',
		'BR', 'EXPD', 'TROW', 'PFG', 'NDAQ', 'BRO', 'ETR', 'ARE', 'MLM', 'TSN',
		'TYL', 'MTD', 'CBOE', 'MCHP', 'COO', 'CINF', 'VRSN', 'SWKS', 'LRCX', 'EXC',
		'PCAR', 'RMD', 'TER', 'CPRT', 'ANSS', 'CTXS', 'ENPH', 'ZBRA', 'FTV', 'IEX',
		'NTAP', 'ADSK', 'TEAM', 'WDAY', 'NOW', 'SNOW', 'PLTR', 'COIN', 'HOOD', 'SOFI',
		'UPST', 'AFRM', 'PYPL', 'SQ', 'V', 'MA', 'AXP', 'DFS', 'COF', 'SYF',
		'FISV', 'FIS', 'GPN', 'FOUR', 'WEX', 'EVTC', 'VRSK', 'INFO', 'MCO', 'SPGI',
		'MSCI', 'NDAQ', 'ICE', 'CME', 'CBOE', 'MKTX', 'TRMB', 'FLT', 'WAT', 'PKI',
		'TDY', 'MKC', 'HSIC', 'ROL', 'BAX', 'AOS', 'BR', 'EXPD', 'TROW', 'PFG',
		'NDAQ', 'BRO', 'ETR', 'ARE', 'MLM', 'TSN', 'TYL', 'MTD', 'CBOE', 'MCHP',
		'COO', 'CINF', 'VRSN', 'SWKS', 'LRCX', 'EXC', 'PCAR', 'RMD', 'TER', 'CPRT',
		'ANSS', 'CTXS', 'ENPH', 'ZBRA', 'FTV', 'IEX', 'NTAP', 'ADSK', 'TEAM', 'WDAY',
		'NOW', 'SNOW', 'PLTR', 'COIN', 'HOOD', 'SOFI', 'UPST', 'AFRM', 'PYPL', 'SQ',
		'V', 'MA', 'AXP', 'DFS', 'COF', 'SYF', 'FISV', 'FIS', 'GPN', 'FOUR',
		'WEX', 'EVTC', 'VRSK', 'INFO', 'MCO', 'SPGI', 'MSCI', 'NDAQ', 'ICE', 'CME',
		'CBOE', 'MKTX', 'TRMB', 'FLT', 'WAT', 'PKI', 'TDY', 'MKC', 'HSIC', 'ROL',
		'BAX', 'AOS', 'BR', 'EXPD', 'TROW', 'PFG', 'NDAQ', 'BRO', 'ETR', 'ARE',
		'MLM', 'TSN', 'TYL', 'MTD', 'CBOE', 'MCHP', 'COO', 'CINF', 'VRSN', 'SWKS',
		'LRCX', 'EXC', 'PCAR', 'RMD', 'TER', 'CPRT', 'ANSS', 'CTXS', 'ENPH', 'ZBRA',
		'FTV', 'IEX', 'NTAP', 'ADSK', 'TEAM', 'WDAY', 'NOW', 'SNOW', 'PLTR', 'COIN',
		'HOOD', 'SOFI', 'UPST', 'AFRM', 'PYPL', 'SQ', 'V', 'MA', 'AXP', 'DFS',
		'COF', 'SYF', 'FISV', 'FIS', 'GPN', 'FOUR', 'WEX', 'EVTC', 'VRSK', 'INFO',
		'MCO', 'SPGI', 'MSCI', 'NDAQ', 'ICE', 'CME', 'CBOE', 'MKTX', 'TRMB', 'FLT',
		'WAT', 'PKI', 'TDY', 'MKC', 'HSIC', 'ROL', 'BAX', 'AOS', 'BR', 'EXPD',
		'TROW', 'PFG', 'NDAQ', 'BRO', 'ETR', 'ARE', 'MLM', 'TSN', 'TYL', 'MTD',
		'CBOE', 'MCHP', 'COO', 'CINF', 'VRSN', 'SWKS', 'LRCX', 'EXC', 'PCAR', 'RMD',
		'TER', 'CPRT', 'ANSS', 'CTXS', 'ENPH', 'ZBRA', 'FTV', 'IEX', 'NTAP'
	]
	return additional


def fetch_prices(
	tickers: Iterable[str],
	start: str,
	end: str,
	save_dir: str,
	period: Optional[str] = None,
	interval: str = "1d",
	auto_adjust: bool = False,
) -> None:
	"""
	Fetch OHLCV for each ticker and save to CSV in save_dir.
	Uses yfinance download per ticker to avoid rate limits.
	"""
	ensure_dir(save_dir)
	start_dt = dt.datetime.fromisoformat(start)
	end_dt = dt.datetime.fromisoformat(end)
	for t in tickers:
		path = os.path.join(save_dir, f"{t}.csv")
		if os.path.exists(path):
			continue
		try:
			df = yf.download(t, start=start_dt, end=end_dt, interval=interval, auto_adjust=auto_adjust, progress=False)
			if df is None or df.empty:
				continue
			
			# Handle MultiIndex columns - flatten them
			if isinstance(df.columns, pd.MultiIndex):
				df.columns = df.columns.get_level_values(0)
			
			# Reset index to make Date a column, then save
			df = df.reset_index()
			df.to_csv(path, index=False)
		except Exception as e:
			print(f"Warning: Could not fetch {t}: {e}")
			continue


def compute_daily_returns_from_csv_dir(
	raw_dir: str,
	start: Optional[str] = None,
	end: Optional[str] = None,
	price_col: str = "Adj Close",
) -> pd.DataFrame:
	"""
	Load per-ticker CSVs (as saved by fetch_prices), compute pct_change on chosen price column,
	and return an aligned DataFrame of returns with columns=tickers, index=datetime.
	"""
	files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
	panels = []
	for f in files:
		path = os.path.join(raw_dir, f)
		try:
			# Read CSV - now Date should be the first column
			df = pd.read_csv(path)
			
			# Set the Date column as index
			df = df.set_index('Date').sort_index()
			df.index = pd.to_datetime(df.index)
			
			if price_col not in df.columns:
				# fallback to Close
				if "Close" in df.columns:
					series = df["Close"].pct_change()
				else:
					continue
			else:
				series = df[price_col].pct_change()
			
			series.name = f.replace(".csv", "")
			panels.append(series)
		except Exception as e:
			print(f"Warning: Could not process {f}: {e}")
			continue
			
	if not panels:
		return pd.DataFrame()
	ret = pd.concat(panels, axis=1).sort_index()
	if start:
		ret = ret[ret.index >= pd.to_datetime(start)]
	if end:
		ret = ret[ret.index <= pd.to_datetime(end)]
	# Drop first row which is NaN due to pct_change per ticker alignment issues
	ret = ret.loc[ret.index[1]:]
	return ret

