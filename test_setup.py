#!/usr/bin/env python3
"""
Test script to verify the framework structure without PyTorch dependencies.
This will help us verify the data fetching and preprocessing work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_fetching():
    """Test data fetching functionality"""
    try:
        from vae_framework.data import get_sp500_tickers, fetch_prices
        print("✓ Data fetching module imported successfully")
        
        # Test getting tickers (limit to 5 for testing)
        tickers = get_sp500_tickers(limit=5)
        print(f"✓ Got {len(tickers)} tickers: {tickers[:3]}...")
        
        return True
    except Exception as e:
        print(f"✗ Data fetching failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing functionality"""
    try:
        from vae_framework.preprocess import filter_panel, compute_standardization
        print("✓ Preprocessing module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return False

def test_dependencies():
    """Test if we can import basic dependencies"""
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        print("✓ Basic dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

if __name__ == "__main__":
    print("Testing VAE Framework Setup...")
    print("=" * 40)
    
    deps_ok = test_dependencies()
    data_ok = test_data_fetching() if deps_ok else False
    preprocess_ok = test_preprocessing() if deps_ok else False
    
    print("=" * 40)
    if deps_ok and data_ok and preprocess_ok:
        print("✓ Framework structure is working!")
        print("\nNext steps:")
        print("1. Install Python 3.10 or 3.11 to get PyTorch support")
        print("2. Or wait for PyTorch to support Python 3.13")
        print("3. Then run: pip install -r requirements.txt")
    else:
        print("✗ Some components need fixing")

