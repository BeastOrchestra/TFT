
import os
import glob
import torch
import numpy as np
import pandas as pd
from itertools import combinations
from torch_geometric_temporal.signal import TemporalData

# ----------------------------
# 1️⃣ Memory-Efficient Stock Data Loading
# ----------------------------
data_dir = "./data/train"  # Directory containing CSV files

selected_features = [
    "close_spy_corr65", "vclose_VIX_corr65", "RSC_VIX_IV", "RSC", "IntradayRange",
    "MACD12269macd", "MACD12269macdsig", "MACD12269macdsigslope", "rsi14", "Close", "vclose"
]

# Get list of stock files
stock_files = glob.glob(os.path.join(data_dir, "*.csv"))
stock_names = [os.path.basename(f).split(".")[0] for f in stock_files]

# Read dates once (to ensure alignment)
sample_df = pd.read_csv(stock_files[0], parse_dates=["date"])
all_dates = sorted(sample_df["date"].unique())

# ----------------------------
# 2️⃣ Define Memory-Efficient Rolling Correlation Calculation
# ----------------------------
def compute_rolling_correlation(stock_files, date_idx):
    """
    Compute rolling correlation for all stock pairs at a given time step,
    loading data stock-by-stock to minimize memory usage.
    """
    rolling_data = []

    for file in stock_files:
        df = pd.read_csv(file, parse_dates=["date"])
        df = df.set_index("date")
        
        # Ensure we only use selected features
        df = df[selected_features]
        
        # Extract past 22 timesteps for correlation calculation
        if all_dates[date_idx] in df.index:
            past_22_days = df.loc[all_dates[date_idx-22:date_idx]].values.flatten()
            rolling_data.append(past_22_days)
        else:
            rolling_data.append(np.full(len(selected_features) * 22, np.nan))  # Placeholder

    rolling_data = np.array(rolling_data)  # Shape: (num_stocks, 22 * num_features)

    # Compute correlation only for valid rows
    valid_rows = ~np.isnan(rolling_data).any(axis=1)
    correlation_matrix = np.corrcoef(rolling_data[valid_rows])

    # Build edge list with normalized weights
    edge_list, edge_weights = [], []

    for (i, j) in combinations(np.where(valid_rows)[0], 2):  # Only consider valid stocks
        weight = (correlation_matrix[i, j] + 1) / 2  # Normalize to [0,1]
        edge_list.append((i, j))
        edge_weights.append(weight)

    return edge_list, edge_weights, valid_rows

# ----------------------------
# 3️⃣ Stream Graph Construction to Avoid Memory Overload
# ----------------------------
temporal_data_list = []

for t in range(22, len(all_dates)):  # Start at 22 to ensure full rolling window
    print(f"Processing timestep {t}/{len(all_dates)}...")

    # Load stocks one at a time, extract rolling features
    x = []
    valid_indices = []

    for i, file in enumerate(stock_files):
        df = pd.read_csv(file, parse_dates=["date"])
        df = df.set_index("date")

        if all_dates[t] in df.index:
            past_22_days = df.loc[all_dates[t-22:t]].values.flatten()
            x.append(past_22_days)
            valid_indices.append(i)
        else:
            x.append(np.full(len(selected_features) * 22, np.nan))  # Placeholder

    x = np.array(x)  # Shape: (num_stocks, 22 * num_features)

    # Compute correlation-based edges while keeping only valid data
    edge_list, edge_weights, valid_rows = compute_rolling_correlation(stock_files, t)

    # Convert to torch tensors (filter valid indices)
    x_tensor = torch.tensor(x[valid_rows], dtype=torch.float)
    edge_index = torch.tensor(edge_list, dtype=torch.long).T  # Shape [2, num_edges]
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)

    # Store temporal snapshot (only for valid data)
    temporal_data = TemporalData(
        x=x_tensor,  # Stock features over past 22 timesteps
        edge_index=edge_index,  # Edges
        edge_attr=edge_attr,  # Edge weights
        edge_timestamp=torch.tensor([t] * len(edge_list), dtype=torch.float)  # Time step per edge
    )

    # Append and save in chunks to avoid memory overload
    temporal_data_list.append(temporal_data)

    if t % 100 == 0:
        torch.save(temporal_data_list, f"tgn_preprocessed_batch_{t}.pt")
        temporal_data_list = []  # Reset to free memory

# Final save
torch.save(temporal_data_list, "tgn_preprocessed_final.pt")
print("Processed TGN data saved in batches.")
