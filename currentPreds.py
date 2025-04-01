import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pandas.errors import EmptyDataError

from torch.utils.data import DataLoader
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import MultiLoss, QuantileLoss, MAE
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch.nn as nn

# -----------------------------
# 1. Device Setup
# -----------------------------
def setup_device():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("medium")  # For NVIDIA Tensor Cores
    print("Using device:", DEVICE)
    return DEVICE

# -----------------------------
# 2. Data Loading Function
# -----------------------------
def load_data(folder):
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            if df.empty:
                print(f"Warning: File {file} is empty; skipping.")
                continue
            df = df.reset_index(drop=True)
            # Drop the 'date' column if it exists
            if 'date' in df.columns:
                df = df.drop('date', axis=1)
            else:
                print(f"Warning: File {file} does not contain a 'date' column.")
            df["time_idx"] = range(len(df))
            df["group"] = os.path.basename(file).split('.')[0]
            df["group"] = df["group"].astype(str)
            # Rename columns: 'Close' -> target_1, 'vclose' -> target_2
            df.rename(columns={"Close": "target_1", "vclose": "target_2"}, inplace=True)
            dfs.append(df)
        except EmptyDataError:
            print(f"EmptyDataError: File {file} is empty or has no columns; skipping.")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        print(f"No data loaded from folder: {folder}")
        return pd.DataFrame()

def load_datasets():
    train_data_folder = "data/train"
    test_data_folder  = "data/test"
    oos_data_folder   = "data/oos"
    train_df = load_data(train_data_folder)
    test_df  = load_data(test_data_folder)
    oos_df   = load_data(oos_data_folder)
    return train_df, test_df, oos_df

# -----------------------------
# 3. Prepare Multi-Target Dataset
# -----------------------------
def prepare_datasets(train_df, test_df, oos_df):
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=["target_1", "target_2"],
        group_ids=["group"],
        max_encoder_length=90,
        max_prediction_length=5,
        static_categoricals=["group"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[
            c for c in train_df.columns if c not in ["group", "time_idx"]
        ],
        target_normalizer=MultiNormalizer([
            TorchNormalizer(method="identity"),
            TorchNormalizer(method="identity")
        ])
    )
    validation = TimeSeriesDataSet.from_dataset(training, test_df, predict_mode=False)
    oos = TimeSeriesDataSet.from_dataset(training, oos_df, predict_mode=False)
    return training, validation, oos

# -----------------------------
# 4. Create DataLoaders
# -----------------------------
def create_dataloaders(training, validation, oos):
    train_dataloader = training.to_dataloader(
        train=True, batch_size=64, shuffle=True, num_workers=4, pin_memory=False
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=32, shuffle=False, num_workers=4, pin_memory=False
    )
    oos_dataloader = oos.to_dataloader(
        train=False, batch_size=16, shuffle=False, num_workers=4, pin_memory=False
    )
    return train_dataloader, val_dataloader, oos_dataloader

# -----------------------------
# 5. Build the Model
# -----------------------------
def build_model(training, learning_rate):
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        lstm_layers=1,
        hidden_size=128,
        attention_head_size=4,
        dropout=0.25,
        hidden_continuous_size=128,
        output_size=[1, 1],  # single output per target for MSE
        loss=MultiLoss([nn.MSELoss(), nn.MSELoss()]),
        log_interval=200,
        reduce_on_plateau_patience=5,
    ).to(DEVICE)
    print(f"Number of params in network: {tft.size() / 1e3:.1f}k")
    return tft

# -----------------------------
# 6. Define LightningModule
# -----------------------------
class TFTLightningModule(LightningModule):
    def __init__(self, tft_model):
        super().__init__()
        self.tft_model = tft_model

    def forward(self, x):
        return self.tft_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        pred = out["prediction"]
        loss = self.tft_model.loss(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        pred = out["prediction"]
        loss = self.tft_model.loss(pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return self.tft_model.configure_optimizers()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        return self(x)

# -----------------------------
# 7. Helper: Move Batch to Device
# -----------------------------
def move_to_device(batch_x, device):
    if isinstance(batch_x, torch.Tensor):
        return batch_x.to(device)
    elif isinstance(batch_x, dict):
        return {k: move_to_device(v, device) for k, v in batch_x.items()}
    elif isinstance(batch_x, list):
        return [move_to_device(item, device) for item in batch_x]
    else:
        return batch_x

# -----------------------------
# 8. Process a Symbol (Generate Predictions)
# -----------------------------
def process_symbol(symbol, training, tft_module, oos_df, mu_sig_df):
    stock_oos_df = oos_df[oos_df["group"] == symbol]
    if stock_oos_df.empty:
        print(f"No OOS data for {symbol}.")
        return None
    try:
        mu_p = mu_sig_df.loc[mu_sig_df['ticker'] == symbol, 'closemu'].values[0]
        sig_p = mu_sig_df.loc[mu_sig_df['ticker'] == symbol, 'closesig'].values[0]
    except Exception as e:
        print(f"Error retrieving mu/sig for {symbol}: {e}")
        return None

    eq_dataset_current = TimeSeriesDataSet.from_dataset(
        training, stock_oos_df, predict_mode=True
    )
    eq_dataloader_current = eq_dataset_current.to_dataloader(
        train=False, batch_size=len(eq_dataset_current), shuffle=False, num_workers=0, pin_memory=False
    )
    eq_dataset_backtest = TimeSeriesDataSet.from_dataset(
        training, stock_oos_df, predict_mode=False
    )
    eq_dataloader_backtest = eq_dataset_backtest.to_dataloader(
        train=False, batch_size=len(eq_dataset_backtest), shuffle=False, num_workers=0, pin_memory=False
    )

    with torch.no_grad():
        for batch in eq_dataloader_current:
            x_current, _ = batch
            x_current = move_to_device(x_current, DEVICE)
            out_current = tft_module(x_current)
            preds_current = out_current["prediction"]
            current_price_preds = preds_current[0][:, :, 0].cpu()
    pred1 = current_price_preds[0, 0].item()
    pred2 = current_price_preds[0, 1].item()
    pred3 = current_price_preds[0, 2].item()
    pred4 = current_price_preds[0, 3].item()
    pred5 = current_price_preds[0, 4].item()

    with torch.no_grad():
        for batch in eq_dataloader_backtest:
            x_backtest, y_tuple = batch
            y_list, _ = y_tuple
            actual_future = y_list[0]
            x_backtest = move_to_device(x_backtest, DEVICE)
            out_backtest = tft_module(x_backtest)
            preds_backtest = out_backtest["prediction"]
            backtest_price_preds = preds_backtest[0][:, :, 0].cpu()
    mse_value = torch.mean((backtest_price_preds[0] - actual_future[0])**2).item()

    result = {
        "Symbol": symbol,
        "MSE": mse_value,
        "Pred1": pred1,
        "Pred2": pred2,
        "Pred3": pred3,
        "Pred4": pred4,
        "Pred5": pred5,
        "Delta": pred5 - pred1,
    }
    return result

# -----------------------------
# Main Execution
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelPath", type=str, default="./models/mar22_model.pth",
                        help="Path to the saved model state dict")
    parser.add_argument("--saveAs", type=str, default="./predictions/RankedPreds_Mar22_Model.csv",
                        help="File path to save predictions")
    parser.add_argument("--learning_rate", type=float, default=3e-05,
                        help="Learning rate for the model")
    args = parser.parse_args()

    global DEVICE
    DEVICE = setup_device()

    train_df, test_df, oos_df = load_datasets()
    training, validation, oos = prepare_datasets(train_df, test_df, oos_df)
    train_dataloader, val_dataloader, oos_dataloader = create_dataloaders(training, validation, oos)

    # Build the model using the provided learning rate
    tft = build_model(training, args.learning_rate)
    tft_module = TFTLightningModule(tft).to(DEVICE)

    # Load saved state dict and set model to evaluation mode
    state_dict = torch.load(args.modelPath, map_location=DEVICE)
    tft_module.load_state_dict(state_dict)
    tft_module.to(DEVICE)
    tft_module.eval()

    # Load scaling parameters from file (mu/sig) 
    mu_sig_df = pd.read_csv('./data/TixMuSig.csv')

    # Process symbols from the oos folder
    results_list = []
    oos_folder = "./data/oos"
    oos_files = [f for f in os.listdir(oos_folder) if f.endswith('.csv')]
    for file in oos_files:
        symbol = os.path.splitext(file)[0].upper()
        print(f"Processing {symbol} ...")
        res = process_symbol(symbol, training, tft_module, oos_df, mu_sig_df)
        if res is not None:
            results_list.append(res)

    results_df = pd.DataFrame(results_list)
    print("\nResults DataFrame:")
    print(results_df)

    results_df_sorted = results_df.sort_values(by="MSE")
    print("\nRanked by MSE:")
    print(results_df_sorted)
    results_df_sorted.to_csv(args.saveAs, index=False)
    
if __name__ == "__main__":
    main()
