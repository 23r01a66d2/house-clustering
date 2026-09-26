"""
Utility functions for logging, directory management, and feature mapping.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import config

# Try configuring stdout to utf-8 if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def ensure_directories():
    """Ensure all required output directories exist."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def log_header(title: str = "HOUSE CLUSTERING SYSTEM"):
    """Print standard system header."""
    print("=" * 60)
    print(f" {title.upper()} ")
    print("=" * 60)

def log_step(step_idx: int, total_steps: int, title: str):
    """Print step progress header."""
    print(f"\n[{step_idx}/{total_steps}] {title}...")

def log_substep(message: str, success: bool = True):
    """Print substep result safely across all console encodings."""
    try:
        symbol = "✓" if success else "!"
        print(f"  {symbol} {message}")
    except (UnicodeEncodeError, Exception):
        symbol = "[OK]" if success else "[!]"
        print(f"  {symbol} {message}")

def log_footer(message: str = "PIPELINE COMPLETED SUCCESSFULLY"):
    """Print pipeline completion banner."""
    print("\n" + "=" * 60)
    print(f" {message.upper()} ")
    print("=" * 60)

def find_column_by_role(df: pd.DataFrame, role: str) -> Optional[str]:
    """
    Dynamically locate the best matching column name in the DataFrame
    for a given semantic feature role (e.g. 'price', 'bedrooms')
    without hardcoding exact column names.
    """
    candidates = config.FEATURE_ROLES.get(role, [])
    # Case 1: Exact lowercase match
    lower_cols = {col.strip().lower(): col for col in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    
    # Case 2: Substring match
    for cand in candidates:
        for col_lower, original_col in lower_cols.items():
            if cand in col_lower:
                return original_col
    return None

def detect_column_roles(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Map all configured feature roles to the actual dataset columns."""
    mapping = {}
    for role in config.FEATURE_ROLES:
        mapping[role] = find_column_by_role(df, role)
    return mapping

def format_number(val, decimals: int = 1) -> str:
    """Format numbers cleanly for display."""
    if pd.isna(val):
        return "N/A"
    if isinstance(val, (int, float)):
        if abs(val) >= 1_000_000:
            return f"{val / 1_000_000:.{decimals}f}M"
        elif abs(val) >= 1_000:
            return f"{val / 1_000:.{decimals}f}K"
        elif isinstance(val, int) or val.is_integer():
            return f"{int(val)}"
        else:
            return f"{val:.{decimals}f}"
    return str(val)

def format_currency(val, decimals: int = 1) -> str:
    """Format Indian Rupee amounts cleanly (e.g. ₹25K, ₹1.5L, ₹1.2Cr)."""
    if pd.isna(val):
        return "N/A"
    val = float(val)
    if abs(val) >= 10_000_000:
        return f"₹{val / 10_000_000:.{decimals}f} Cr"
    elif abs(val) >= 100_000:
        return f"₹{val / 100_000:.{decimals}f} L"
    elif abs(val) >= 1_000:
        return f"₹{val / 1_000:.0f}K"
    else:
        return f"₹{val:,.0f}"
