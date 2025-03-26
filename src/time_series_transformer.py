#unfinished/ still expperimenting
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import os
from utils import add_features
import itertools
from sklearn.metrics import mean_squared_error
import json
from datetime import datetime

##############################################
# Time-Based Split Function
##############################################
def time_based_split(dataset, val_ratio=0.2):
    """
    Split dataset chronologically to prevent data leakage.
    
    Args:
        dataset: The dataset to split
        val_ratio: Ratio of data to use for validation
        
    Returns:
        train_dataset, val_dataset: The split datasets
    """
    split_idx = int(len(dataset) * (1 - val_ratio))
    train_dataset = torch.utils.data.Subset(dataset, range(0, split_idx))
    val_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
    return train_dataset, val_dataset

##############################################
# Grid Search for Hyperparameter Optimization
##############################################
def grid_search_hyperparameters(csv_file, param_grid, seq_len=30, k_best_features=10, 
                               val_split=0.2, batch_size=32, epochs=20, early_stopping_patience=5):
    """
    Perform grid search to find the best hyperparameters for the TimeSeriesTransformer model.
    
    Args:
        csv_file (str): Path to the CSV file with stock data
        param_grid (dict): Dictionary of hyperparameter names and possible values
        seq_len (int): Sequence length for the model
        k_best_features (int): Number of best features to select
        val_split (float): Validation split ratio
        batch_size (int): Batch size for training
        epochs (int): Maximum number of training epochs
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping
    
    Returns:
        dict: Best hyperparameters and their performance
    """
    print(f"Starting grid search on {csv_file}...")
    
    # Create dataset with feature selection - use train_only=True to prevent data leakage
    dataset = StockDataset(csv_file, seq_len=seq_len, k_best_features=k_best_features, 
                          train_only=True, train_ratio=(1-val_split))
    
    # Split into train and validation sets chronologically
    train_dataset, val_dataset = time_based_split(dataset, val_split)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for time series
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input dimension from dataset
    input_dim = len(dataset.feature_list)
    
    # Generate all combinations of hyperparameters
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    hyperparameter_combinations = list(itertools.product(*values))
    
    # Store results
    results = []
    
    # Create directory for grid search results
    os.makedirs('grid_search_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'grid_search_results/grid_search_{timestamp}.json'
    
    print(f"Total combinations to try: {len(hyperparameter_combinations)}")
    
    # Try each combination
    for i, combination in enumerate(hyperparameter_combinations):
        hyperparams = dict(zip(keys, combination))
        print(f"\nTrying combination {i+1}/{len(hyperparameter_combinations)}:")
        for k, v in hyperparams.items():
            print(f"  {k}: {v}")
        
        # Create model with current hyperparameters
        model = TimeSeriesTransformer(
            input_dim=input_dim, 
            model_dim=hyperparams.get('model_dim', 64), 
            num_heads=hyperparams.get('num_heads', 4), 
            num_layers=hyperparams.get('num_layers', 2), 
            dropout=hyperparams.get('dropout', 0.2)
        )
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        lr = hyperparams.get('lr', 1e-3)
        weight_decay = hyperparams.get('weight_decay', 1e-5)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Training with early stopping
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_train_losses = []
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())
            
            avg_train_loss = np.mean(epoch_train_losses)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            epoch_val_losses = []
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    epoch_val_losses.append(loss.item())
                    
                    # Store predictions and targets for metrics
                    val_predictions.extend(outputs.squeeze().tolist())
                    val_targets.extend(batch_y.squeeze().tolist())
            
            avg_val_loss = np.mean(epoch_val_losses)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Check if this is the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Calculate additional metrics on validation set
        mse = mean_squared_error(val_targets, val_predictions)
        
        # Calculate direction accuracy
        direction_correct = sum(1 for p, t in zip(val_predictions, val_targets) 
                               if (p > 0 and t > 0) or (p < 0 and t < 0))
        direction_accuracy = direction_correct / len(val_predictions) if val_predictions else 0
        
        # Store results for this combination
        result = {
            'hyperparameters': hyperparams,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'mse': mse,
            'direction_accuracy': direction_accuracy,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        results.append(result)
        
        # Save results after each combination
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {results_file}")
        print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
        print(f"MSE: {mse:.6f}, Direction Accuracy: {direction_accuracy:.4f}")
    
    # Find the best combination
    best_result = min(results, key=lambda x: x['best_val_loss'])
    
    print("\n=== Grid Search Complete ===")
    print("Best hyperparameters:")
    for k, v in best_result['hyperparameters'].items():
        print(f"  {k}: {v}")
    print(f"Best validation loss: {best_result['best_val_loss']:.6f}")
    print(f"MSE: {best_result['mse']:.6f}, Direction Accuracy: {best_result['direction_accuracy']:.4f}")
    
    return best_result

##############################################
# Dataset Definition with Feature Selection
##############################################
class StockDataset(Dataset):
    def __init__(self, csv_file, seq_len=30, feature_list=None, k_best_features=10, 
                train_only=False, train_ratio=0.8):
        """
        Args:
            csv_file (str): Path to the CSV file.
            seq_len (int): Length of the input sequence.
            feature_list (list): List of feature column names to use.
            k_best_features (int): Number of best features to select.
            train_only (bool): Whether to use only training data for feature selection.
            train_ratio (float): Ratio of data to use for training.
        """
        self.seq_len = seq_len
        
        # Read CSV file
        try:
            data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            print(f"Data loaded with DatetimeIndex: {data.index[0]} to {data.index[-1]}")
        except:
            data = pd.read_csv(csv_file)
            # Check for date column
            date_column = None
            for col in ['Date', 'date', 'datetime', 'Datetime']:
                if col in data.columns:
                    date_column = col
                    break
            
            if date_column:
                data[date_column] = pd.to_datetime(data[date_column])
                data.set_index(date_column, inplace=True)
                print(f"Data loaded with date column: {data.index[0]} to {data.index[-1]}")
            else:
                print(f"Warning: No date column found. Available columns: {data.columns.tolist()}")
        
        # Apply engineered features
        data = add_features(data)
        print(f"Data shape after feature engineering: {data.shape}")
        
        # Define target as next day's normalized return (clipped between -1 and 1)
        close_prices = data['Close'].values.astype(np.float32)
        returns = (np.roll(close_prices, -1) - close_prices) / close_prices
        returns = np.clip(returns, -1, 1)
        
        # Remove last sample because target isn't available for it
        returns = returns[:-1]
        data = data.iloc[:-1]
        
        # Scientific Feature Selection
        # 1. Remove features with too many missing values
        missing_threshold = 0.05  # 5% missing values
        data_features = data.select_dtypes(include=[np.number])
        missing_ratio = data_features.isnull().mean()
        valid_features = missing_ratio[missing_ratio < missing_threshold].index.tolist()
        
        # 2. Fill remaining missing values
        data_features = data_features[valid_features]
        data_features = data_features.ffill().bfill()  # Forward fill then backward fill
        
        # 3. Apply feature selection using F-regression
        # If train_only is True, only use training portion for feature selection
        if train_only:
            train_size = int(len(data) * train_ratio)
            print(f"Using only first {train_size} samples for feature selection to prevent data leakage")
            X_train = data_features.iloc[:train_size].values
            y_train = returns[:train_size]
            
            # Standardize features using only training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Select k best features using only training data
            selector = SelectKBest(f_regression, k=min(k_best_features, len(valid_features)))
            selector.fit(X_train_scaled, y_train)
            
            # Get selected feature indices and names
            selected_indices = selector.get_support(indices=True)
            self.feature_list = [data_features.columns[i] for i in selected_indices]
            
            # Apply the same scaler and feature selection to the entire dataset
            X_all_scaled = scaler.transform(data_features.values)
            self.data = X_all_scaled[:, selected_indices].astype(np.float32)
        else:
            # Use all data for feature selection (only for testing/debugging)
            X = data_features.values
            y = returns
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Select k best features
            selector = SelectKBest(f_regression, k=min(k_best_features, len(valid_features)))
            selector.fit(X_scaled, y)
            
            # Get selected feature indices and names
            selected_indices = selector.get_support(indices=True)
            self.feature_list = [data_features.columns[i] for i in selected_indices]
            
            # Convert selected features to a NumPy array
            self.data = X_scaled[:, selected_indices].astype(np.float32)
        
        print(f"Selected {len(self.feature_list)} best features: {self.feature_list}")
        self.targets = returns
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        # Get a sequence of features and the corresponding target value.
        x = self.data[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x), torch.tensor([y])

##############################################
# 3. Time Series Transformer Model
##############################################
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2, dropout=0.2):
        """
        Args:
            input_dim (int): Number of features per time step.
            model_dim (int): Dimension of the model embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate for regularization.
        """
        super(TimeSeriesTransformer, self).__init__()
         # Add positional encoding to capture time information better
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, model_dim))  # Max 100 time steps
        # Project input features to the model dimension
        self.input_projection = nn.Linear(input_dim, model_dim)
        # Build the Transformer Encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True  # Set batch_first=True for better performance
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Add more layers for better feature extraction
        self.fc1 = nn.Linear(model_dim, model_dim // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(model_dim // 2, 1)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, seq_length, input_dim)
            
        Returns:
            torch.Tensor: Prediction of shape (batch_size, 1) in range (-1, 1)
        """
        seq_len = x.size(1)
        x = self.input_projection(x)  # (batch_size, seq_length, model_dim)
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.transformer_encoder(x)  # batch_first=True so no transpose needed
        out = x[:, -1, :]  # Use the output at the last time step
        # Additional layers
        out = torch.nn.functional.relu(self.fc1(out))
        out = self.dropout1(out)
        out = self.fc2(out)
        return torch.tanh(out)  # Squash the output to (-1, 1)

##############################################
# 4. Signal Generation with Confidence Score
##############################################
def generate_signal_with_confidence(prediction, threshold=0.002):
    """
    Convert the continuous prediction into a discrete signal and confidence score.
    
    Args:
        prediction (float): Model output in the range [-1, 1].
        threshold (float): Threshold for signal generation.

    Returns:
        signal (int): 1 for buy, -1 for sell, 0 for hold.
        confidence (float): Normalized confidence score in [0, 1].
    """
    abs_pred = abs(prediction)
    if prediction > threshold:
        signal = 1
    elif prediction < -threshold:
        signal = -1
    else:
        signal = 0
    
    # Confidence is calculated as how far the prediction is from the threshold,
    # normalized to the maximum possible distance (1 - threshold).
    if abs_pred > threshold:
        confidence = (abs_pred - threshold) / (1 - threshold)
        confidence = min(confidence, 1.0)
    else:
        confidence = 0.0
    
    return signal, confidence

##############################################
# 5. Training Loop with Improved Monitoring
##############################################
def train_model(csv_file, epochs=20, seq_len=30, batch_size=32, lr=1e-3, 
               model_dim=64, num_heads=4, num_layers=2, dropout=0.2,
               weight_decay=1e-5, k_best_features=10, val_split=0.2,
               early_stopping_patience=5, verbose=True):
    """
    Train the transformer model with improved monitoring and early stopping.
    
    Args:
        csv_file (str): Path to the CSV file
        epochs (int): Number of training epochs
        seq_len (int): Sequence length
        batch_size (int): Batch size
        lr (float): Learning rate
        model_dim (int): Model dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dropout (float): Dropout rate
        weight_decay (float): Weight decay for regularization
        k_best_features (int): Number of best features to select
        val_split (float): Validation split ratio
        early_stopping_patience (int): Patience for early stopping
        verbose (bool): Whether to print detailed logs
        
    Returns:
        tuple: Trained model, feature list, and training history
    """
    if verbose:
        print(f"Training model on {csv_file}...")
    
    # Create dataset with feature selection - use train_only=True to prevent data leakage
    dataset = StockDataset(csv_file, seq_len=seq_len, k_best_features=k_best_features, 
                          train_only=True, train_ratio=(1-val_split))
    
    # Split into train and validation sets chronologically
    train_dataset, val_dataset = time_based_split(dataset, val_split)
    
    # No shuffling for time series data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input dimension from dataset
    input_dim = len(dataset.feature_list)
    feature_list = dataset.feature_list
    
    if verbose:
        print(f"Using {len(feature_list)} features: {feature_list}")
        print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    # Create model
    model = TimeSeriesTransformer(
        input_dim=input_dim, 
        model_dim=model_dim, 
        num_heads=num_heads, 
        num_layers=num_layers, 
        dropout=dropout
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # For tracking training progress
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
                
                # Store predictions and targets for metrics
                val_predictions.extend(outputs.squeeze().tolist())
                val_targets.extend(batch_y.squeeze().tolist())
        
        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")
        
        # Learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Check if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
            if verbose:
                print(f"New best model saved with validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Calculate additional metrics on validation set
    model.eval()
    with torch.no_grad():
        val_predictions = []
        val_targets = []
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            val_predictions.extend(outputs.squeeze().tolist())
            val_targets.extend(batch_y.squeeze().tolist())
    
    # Calculate direction accuracy
    direction_correct = sum(1 for p, t in zip(val_predictions, val_targets) 
                           if (p > 0 and t > 0) or (p < 0 and t < 0))
    direction_accuracy = direction_correct / len(val_predictions) if val_predictions else 0
    
    # Calculate signal distribution with a default threshold
    threshold = 0.03
    signals = [generate_signal_with_confidence(p, threshold)[0] for p in val_predictions]
    buy_signals = signals.count(1)
    sell_signals = signals.count(-1)
    hold_signals = signals.count(0)
    
    if verbose:
        print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
        print(f"Direction accuracy: {direction_accuracy:.4f}")
        print(f"Signal distribution: BUY: {buy_signals}, SELL: {sell_signals}, HOLD: {hold_signals}")
    
    # Add final metrics to history
    history['best_val_loss'] = best_val_loss
    history['direction_accuracy'] = direction_accuracy
    history['signal_distribution'] = {
        'buy': buy_signals,
        'sell': sell_signals,
        'hold': hold_signals
    }
    
    return model, feature_list, history

##############################################
# 6. Main: Run Grid Search and Train Best Model
##############################################
if __name__ == "__main__":
    ticker = "AAPL"
    data_dir = "data"
    csv_file = os.path.join(data_dir, f"{ticker}.csv")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in data directory: {os.listdir(data_dir) if os.path.exists(data_dir) else 'data directory not found'}")
        exit(1)
    
    # Define parameter grid for grid search
    param_grid = {
        'model_dim': [32, 64, 128],
        'num_heads': [2, 4, 8],
        'num_layers': [1, 2, 3],
        'dropout': [0.1, 0.2, 0.3],
        'lr': [1e-4, 1e-3, 5e-3],
        'weight_decay': [1e-6, 1e-5, 1e-4]
    }
    
    # Ask user if they want to run grid search (it can take a long time)
    run_grid_search = input("Run grid search? This can take a long time. (y/n): ").lower() == 'y'
    
    if run_grid_search:
        # Run grid search with a subset of parameters to save time
        reduced_param_grid = {
            'model_dim': [64],
            'num_heads': [4],
            'num_layers': [1, 2],
            'dropout': [0.2],
            'lr': [1e-4, 1e-3],
            'weight_decay': [1e-5]
        }
        
        best_params = grid_search_hyperparameters(
            csv_file=csv_file,
            param_grid=reduced_param_grid,
            seq_len=30,
            k_best_features=10,
            val_split=0.2,
            batch_size=32,
            epochs=10,  # Reduced epochs for grid search
            early_stopping_patience=3
        )
        
        print("\nBest parameters found:")
        for k, v in best_params['hyperparameters'].items():
            print(f"  {k}: {v}")
        
        # Train final model with best parameters
        best_hyperparams = best_params['hyperparameters']
        model, feature_list, history = train_model(
            csv_file=csv_file,
            epochs=20,
            seq_len=30,
            batch_size=32,
            lr=best_hyperparams.get('lr', 1e-3),
            model_dim=best_hyperparams.get('model_dim', 64),
            num_heads=best_hyperparams.get('num_heads', 4),
            num_layers=best_hyperparams.get('num_layers', 2),
            dropout=best_hyperparams.get('dropout', 0.2),
            weight_decay=best_hyperparams.get('weight_decay', 1e-5),
            k_best_features=10,
            val_split=0.2,
            early_stopping_patience=5
        )
    else:
        # Train model with default parameters
        model, feature_list, history = train_model(
            csv_file=csv_file,
            epochs=20,
            seq_len=30,
            batch_size=32,
            lr=1e-3,
            model_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.2,
            weight_decay=1e-5,
            k_best_features=10,
            val_split=0.2,
            early_stopping_patience=5
        )
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_list': feature_list,
        'history': history
    }, f"models/{ticker}_transformer_model.pt")
    
    print(f"Model saved to models/{ticker}_transformer_model.pt")
    
    # Test the model on a few samples from the validation set
    dataset = StockDataset(csv_file, seq_len=30, k_best_features=10, train_only=True)
    _, val_dataset = time_based_split(dataset, 0.2)
    
    # Test on a few samples
    model.eval()
    with torch.no_grad():
        print("\nTesting model on validation samples:")
        for i in range(min(5, len(val_dataset))):
            sample_x, sample_y = val_dataset[i]
            prediction = model(sample_x.unsqueeze(0))
            pred_value = prediction.item()
            signal, confidence = generate_signal_with_confidence(pred_value, threshold=0.03)
            signal_type = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
            
            print(f"Sample {i}:")
            print(f"  Prediction: {pred_value:.6f}")
            print(f"  Signal: {signal_type} (confidence: {confidence:.4f})")
            print(f"  Actual target: {sample_y.item():.6f}")
            print(f"  Direction match: {'Yes' if np.sign(pred_value) == np.sign(sample_y.item()) else 'No'}")
    
    # Plot training history
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['learning_rates'])
        plt.title('Learning Rate During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{ticker}_training_history.png")
        plt.close()
        
        # Plot signal distribution
        signal_dist = history['signal_distribution']
        plt.figure(figsize=(8, 6))
        plt.bar(['Buy', 'Hold', 'Sell'], [signal_dist['buy'], signal_dist['hold'], signal_dist['sell']])
        plt.title('Signal Distribution on Validation Set')
        plt.ylabel('Count')
        plt.savefig(f"plots/{ticker}_signal_distribution.png")
        plt.close()
        
        print(f"Training plots saved to plots/{ticker}_training_history.png and plots/{ticker}_signal_distribution.png")
    except ImportError:
        print("Matplotlib not available for plotting.")
    except Exception as e:
        print(f"Error creating plots: {e}")

##############################################
# 7. Walk-Forward Validation
##############################################
def walk_forward_validation(csv_file, seq_len=30, k_best_features=10, 
                           window_size=252, step_size=21, 
                           model_dim=64, num_heads=4, num_layers=2, dropout=0.2,
                           lr=1e-3, weight_decay=1e-5, epochs=10, batch_size=32):
    """
    Perform walk-forward validation to test the model's performance over time.
    
    Args:
        csv_file (str): Path to the CSV file
        seq_len (int): Sequence length
        k_best_features (int): Number of best features to select
        window_size (int): Size of the training window in days
        step_size (int): Number of days to move forward in each step
        model_dim (int): Model dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dropout (float): Dropout rate
        lr (float): Learning rate
        weight_decay (float): Weight decay for regularization
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        
    Returns:
        dict: Walk-forward validation results
    """
    print(f"Starting walk-forward validation on {csv_file}...")
    
    # Read data
    data = pd.read_csv(csv_file)
    
    # Check if 'Date' column exists
    date_column = None
    if 'Date' in data.columns:
        date_column = 'Date'
    elif 'date' in data.columns:
        date_column = 'date'
    
    if date_column:
        data[date_column] = pd.to_datetime(data[date_column])
        data.sort_values(date_column, inplace=True)
    
    # Apply feature engineering
    data = add_features(data)
    
    # Ensure we have enough data
    if len(data) < window_size + step_size:
        print(f"Not enough data for walk-forward validation. Need at least {window_size + step_size} samples.")
        return None
    
    # Initialize results
    results = {
        'dates': [],
        'predictions': [],
        'actual_returns': [],
        'signals': [],
        'confidences': [],
        'direction_accuracy': [],
        'mse': []
    }
    
    # Walk forward through the data
    for start_idx in range(0, len(data) - window_size - step_size, step_size):
        end_idx = start_idx + window_size
        test_end_idx = end_idx + step_size
        
        print(f"\nTraining on data from index {start_idx} to {end_idx-1}")
        print(f"Testing on data from index {end_idx} to {test_end_idx-1}")
        
        if date_column:
            train_start_date = data.iloc[start_idx][date_column]
            train_end_date = data.iloc[end_idx-1][date_column]
            test_start_date = data.iloc[end_idx][date_column]
            test_end_date = data.iloc[min(test_end_idx-1, len(data)-1)][date_column]
            print(f"Training period: {train_start_date} to {train_end_date}")
            print(f"Testing period: {test_start_date} to {test_end_date}")
        
        # Extract training and testing data
        train_data = data.iloc[start_idx:end_idx].copy()
        test_data = data.iloc[end_idx:test_end_idx].copy()
        
        # Save to temporary CSV files
        train_file = "temp_train.csv"
        test_file = "temp_test.csv"
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        # Train model on this window
        model, feature_list, _ = train_model(
            csv_file=train_file,
            epochs=epochs,
            seq_len=seq_len,
            batch_size=batch_size,
            lr=lr,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            weight_decay=weight_decay,
            k_best_features=k_best_features,
            val_split=0.2,
            early_stopping_patience=3,
            verbose=False
        )
        
        # Create dataset for testing
        test_dataset = StockDataset(test_file, seq_len=seq_len, feature_list=feature_list)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Generate predictions
        model.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                test_predictions.extend(outputs.squeeze().tolist())
                test_targets.extend(batch_y.squeeze().tolist())
        
        # Generate signals
        threshold = 0.03
        signals = []
        confidences = []
        
        for pred in test_predictions:
            signal, confidence = generate_signal_with_confidence(pred, threshold)
            signals.append(signal)
            confidences.append(confidence)
        
        # Calculate metrics
        direction_correct = sum(1 for p, t in zip(test_predictions, test_targets) 
                               if (p > 0 and t > 0) or (p < 0 and t < 0))
        direction_accuracy = direction_correct / len(test_predictions) if test_predictions else 0
        
        mse = mean_squared_error(test_targets, test_predictions) if test_predictions else 0
        
        # Store results
        if date_column:
            results['dates'].extend(test_data[date_column].tolist())
        else:
            results['dates'].extend(list(range(end_idx, test_end_idx)))
        
        results['predictions'].extend(test_predictions)
        results['actual_returns'].extend(test_targets)
        results['signals'].extend(signals)
        results['confidences'].extend(confidences)
        results['direction_accuracy'].append(direction_accuracy)
        results['mse'].append(mse)
        
        print(f"Direction accuracy: {direction_accuracy:.4f}, MSE: {mse:.6f}")
        print(f"Signal distribution: BUY: {signals.count(1)}, SELL: {signals.count(-1)}, HOLD: {signals.count(0)}")
        
        # Clean up temporary files
        os.remove(train_file)
        os.remove(test_file)
    
    # Calculate overall metrics
    overall_direction_accuracy = sum(1 for p, t in zip(results['predictions'], results['actual_returns']) 
                                   if (p > 0 and t > 0) or (p < 0 and t < 0))
    overall_direction_accuracy /= len(results['predictions']) if results['predictions'] else 1
    
    overall_mse = mean_squared_error(results['actual_returns'], results['predictions']) if results['predictions'] else 0
    
    print("\nWalk-Forward Validation Complete")
    print(f"Overall Direction Accuracy: {overall_direction_accuracy:.4f}")
    print(f"Overall MSE: {overall_mse:.6f}")
    print(f"Total Predictions: {len(results['predictions'])}")
    print(f"Signal Distribution: BUY: {results['signals'].count(1)}, SELL: {results['signals'].count(-1)}, HOLD: {results['signals'].count(0)}")
    
    # Save results
    results_df = pd.DataFrame({
        'Date': results['dates'],
        'Prediction': results['predictions'],
        'Actual_Return': results['actual_returns'],
        'Signal': results['signals'],
        'Confidence': results['confidences']
    })
    
    os.makedirs("results", exist_ok=True)
    results_file = f"results/walk_forward_{os.path.basename(csv_file).split('.')[0]}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot predictions vs actual returns
        plt.subplot(2, 1, 1)
        plt.plot(results['dates'], results['predictions'], label='Predictions', alpha=0.7)
        plt.plot(results['dates'], results['actual_returns'], label='Actual Returns', alpha=0.7)
        plt.title('Predictions vs Actual Returns')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot direction accuracy over time
        plt.subplot(2, 1, 2)
        window_indices = list(range(0, len(results['dates']), step_size))
        plt.plot(window_indices, results['direction_accuracy'], marker='o')
        plt.title('Direction Accuracy by Window')
        plt.xlabel('Window Index')
        plt.ylabel('Direction Accuracy')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"plots/walk_forward_{os.path.basename(csv_file).split('.')[0]}.png")
        plt.close()
        
        print(f"Plot saved to plots/walk_forward_{os.path.basename(csv_file).split('.')[0]}.png")
    except ImportError:
        print("Matplotlib not available for plotting.")
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    return results


        
