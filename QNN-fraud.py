# AI DISCLOSURE: we used AI to help structure and debug this code. Then AI was 
# used to generate explanatory comments throughout.

# =============================================================================
# --- SCRIPT: ADVANCED HYBRID QUANTUM CLASSIFIER EXPERIMENT ---
#
# This script provides a full framework for developing, training, and rigorously
# evaluating a hybrid quantum-classical neural network against its classical
# counterpart on the task of credit card fraud detection.
#
# 1. A central CONFIG section to easily manage all hyperparameters.
# 2. Automated creation of unique, timestamped directories for each run
#    to save all important artifacts (model weights, data transformers, etc.).
# 3. A master log file ('master_log_maxpairs.txt') that is appended with the key
#    results of every model trained, allowing for long-term analysis.
# 4. Detailed per-epoch training and validation metrics printed to the console
#    to monitor learning progress.
# 5. Advanced quantum circuit design, including a data-inspired entanglement
#    scheme based on classical feature correlation analysis.
# 6. Comprehensive final evaluation with business-centric metrics (PR-AUC)
#    and visualizations (Gains Charts).
# 7. A final summary of the model architectures for clear comparison.
# =============================================================================


# =============================================================================
# SECTION 1: IMPORTS AND SETUP
# =============================================================================
# --- Standard Libraries for file handling, time, math, and data structures ---
import time
import pickle
from datetime import datetime
from pathlib import Path

# --- Core Data Science and Plotting Libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- PyTorch Libraries for building and training neural networks ---
import torch
import torch.nn as nn
import torchinfo
from torch.utils.data import DataLoader, TensorDataset

# --- Scikit-learn for data processing and a rich set of performance metrics ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, auc
)

# --- PennyLane for creating and simulating Quantum Machine Learning models ---
import pennylane as qml

# --- Utility for creating progress bars ---
from tqdm import tqdm

# =============================================================================
# SECTION 2: CENTRAL CONFIGURATION
# =============================================================================
# This dictionary acts as the main control panel for the entire experiment.
# Modifying a value here will propagate it throughout the script.
CONFIG = {
    # --- Data and Splitting Controls ---
    "sample_size": 10000,    # Number of normal (non-fraud) transactions to sample.
    "train_split": 0.5,     # Percentage of data for the training set (e.g., 0.5 for 50%).
    "val_split": 0.25,       # Percentage of data for the validation set (e.g., 0.25 for 25%).
                            # The test split will be automatically calculated as 1.0 - train_split - val_split.

    # --- Quantum Circuit Controls ---
    "n_qubits": 10,          # Number of qubits in the quantum circuit. This is a key hyperparameter.
    "num_q_layers": 1,      # Number of trainable quantum layers (depth) per data re-upload.
    "num_priority_pairs": 0,# Number of top correlated feature pairs to explicitly entangle.

    # --- Model Training Controls ---
    "threshold": 0.3,       # Classification threshold for calculating metrics like precision/recall.
    "epochs": 10,           # Number of complete passes through the training dataset.
    "learning_rate": 0.001,  # Learning rate for the Adam optimizer.
}


# =============================================================================
# SECTION 3: PRE-DECLARATION OF GLOBAL VARIABLES
# =============================================================================
# These variables are defined in the global scope so they can be set within the
# main() function but still be accessible to the globally-defined QNode decorator.
# This is a clean way to manage dynamic circuit configurations in a script.

# This map will store which qubits to entangle based on correlation analysis.
priority_entanglement_map = []
# The quantum device is defined once, using the number of qubits from the CONFIG.
dev = qml.device("default.qubit", wires=CONFIG["n_qubits"])


# =============================================================================
# SECTION 4: HYBRID MODEL ARCHITECTURE AND QUANTUM CIRCUIT
# =============================================================================
# The @qml.qnode decorator transforms a Python function into a quantum circuit
# that can be executed on the specified device ('dev').
@qml.qnode(dev, interface="torch", diff_method="backprop")
def custom_quantum_circuit(inputs, weights):
    """
    Defines the quantum circuit with data re-uploading and a custom entanglement strategy.
    The 'backprop' differentiation method allows it to integrate seamlessly with PyTorch's autograd.
    
    Args:
        inputs (torch.Tensor): The preprocessed and encoded features from the classical layers.
        weights (torch.Tensor): The trainable parameters/weights for the quantum gates.
    """
    # Get key parameters from the global CONFIG.
    n_qubits = CONFIG["n_qubits"]
    
    # Calculate the number of chunks needed to load all features onto the qubits.
    num_chunks = int(np.ceil(inputs.shape[-1] / n_qubits))

    # This is the data re-uploading loop.
    for i in range(num_chunks):
        # Select a "chunk" of features to load in this iteration.
        feature_chunk = inputs[..., i*n_qubits:(i+1)*n_qubits]
        
        # If the last chunk is smaller than the number of qubits, pad it with zeros.
        if feature_chunk.shape[-1] < n_qubits:
            padding = torch.zeros(inputs.shape[:-1] + (n_qubits - feature_chunk.shape[-1],), device=inputs.device)
            feature_chunk = torch.cat([feature_chunk, padding], dim=-1)
            
        # Encode the classical feature chunk into the quantum state by rotating the qubits.
        qml.AngleEmbedding(features=feature_chunk, wires=range(n_qubits), rotation='Y')

        # --- CUSTOM ENTANGLEMENT STAGE ---
        # For the current chunk 'i', check our globally defined map for any priority pairs to entangle.
        # This injects our prior knowledge from the correlation analysis directly into the circuit.
        for item in priority_entanglement_map:
            if item['chunk'] == i:
            # REPLACE THIS:
                # qml.CNOT(wires=item['qubits'])

            # WITH THIS:
                qml.CNOT(wires=item['qubits'])
        # --------------------------------
        
        qml.StronglyEntanglingLayers(weights[i], wires=range(n_qubits))

        # Apply a standard trainable layer of rotations and entanglement.
        # The depth of this layer is controlled by `num_q_layers` in the CONFIG.


    # Apply a final, all-to-all entanglement layer to maximize global information mixing
#     # before measurement.
#     for i in range(n_qubits):
#         for j in range(i + 1, n_qubits):
#             qml.CNOT(wires=[i, j])

    # Return the expectation value of the Pauli-Z operator for each qubit.
    # This collapses the quantum state into a classical vector of length `n_qubits`.
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridModel(nn.Module):
    """
    A redesigned, symmetric hybrid model for a fairer comparison.
    Both classical and quantum paths share an identical feature extraction trunk.
    """
    def __init__(self, n_features, config, qnode=None):
        super().__init__()
        self.use_quantum = (qnode is not None)
        nq = config["n_qubits"]

        # --- Shared Feature Extraction Trunk ---
        # This part is identical for both the classical and hybrid models.
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.Tanh()
#             nn.LayerNorm(n_features) # MADE SLIMMER WITH THIS
        )

        # --- Interchangeable Middle Layer ---
        # This is the only part of the feature processing that differs.
        if self.use_quantum:
            # The quantum processing block.
            num_chunks = int(np.ceil(n_features / nq))
            weight_shapes = {"weights": (num_chunks, config["num_q_layers"], nq, 3)}
            self.middle_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
            # Post-processing to match the classical path (does nothing here).
            self.post_processing = nn.Identity()
        else:
            # The classical processing block, designed to mirror the quantum layer's function.
            self.middle_layer = nn.Linear(n_features, nq)
            # Post-processing to match the quantum output range [-1, 1].
            self.post_processing = nn.Tanh()

        # --- Shared Classification Head ---
        # This final layer is also identical for both models.
        self.head = nn.Linear(nq, 1)

    def forward(self, x):
        # 1. Pass data through the shared feature extraction trunk.
        x = self.feature_extractor(x)
        
        # 2. Pass data through the interchangeable middle layer (quantum or classical).
        x = self.middle_layer(x)
        
        # 3. Apply the symmetric post-processing.
        x = self.post_processing(x)
        
        # 4. Pass data through the final classification head to get the logit.
        return self.head(x)

# =============================================================================
# SECTION 5: MAIN EXPERIMENT EXECUTION FUNCTION
# =============================================================================
def main():
    """Main function to orchestrate the entire experiment from start to finish."""
    global priority_entanglement_map # Declare that we will modify the global variable

    # --- Setup a unique, timestamped directory for this run's outputs ---
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("outputs") / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory for this run: {output_dir}")

    # --- Save the configuration used for this run for perfect reproducibility ---
    with open(output_dir / "config.txt", "w") as f:
        for key, value in CONFIG.items(): f.write(f"{key}: {value}\n")

    # --- Data Preparation ---
    print("\n--- Starting Data Preparation ---")
    df = pd.read_csv('creditcard.csv')
    normal_transactions = df[df['Class'] == 0].sample(n=CONFIG["sample_size"], random_state=42)
    fraud_transactions = df[df['Class'] == 1]
    df_reduced = pd.concat([normal_transactions, fraud_transactions]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_reduced['hour_of_the_day'] = (df_reduced['Time'] / 3600) % 24; df_reduced = df_reduced.drop(['Time'], axis=1)
    X = df_reduced.drop('Class', axis=1).values; y = df_reduced['Class'].values
    original_feature_names = [f'V{i+1}' for i in range(X.shape[1]-1)] + ['hour_of_the_day']

    # --- Correlation Analysis and Feature Reordering ---
    # --- Data Splitting using values from CONFIG (moved BEFORE correlation to avoid leakage) ---
    test_and_val_size = 1.0 - CONFIG["train_split"]
    test_split_relative = (1.0 - CONFIG["train_split"] - CONFIG["val_split"]) / test_and_val_size if test_and_val_size > 0 else 0
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_and_val_size, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_split_relative, random_state=42, stratify=y_temp)

    # --- Correlation Analysis and Feature Reordering (TRAIN-ONLY) ---
    # Compute correlations on TRAIN ONLY to prevent leakage
    df_corr_train = pd.DataFrame(X_train, columns=original_feature_names)
    corr = df_corr_train.corr().abs()

    # Use UPPER TRIANGLE ONLY to avoid duplicates like (A,B) and (B,A)
    iu = np.triu_indices_from(corr, k=1)
    pairs_series = pd.Series(
        corr.values[iu],
        index=pd.MultiIndex.from_arrays([corr.index.values[iu[0]], corr.columns.values[iu[1]]])
    ).sort_values(ascending=False)

    # Greedy feature order from top correlated pairs, then leftovers
    new_feature_order, seen_features = [], set()
    for (f1, f2), _ in pairs_series.items():
        if f1 not in seen_features and f2 not in seen_features:
            new_feature_order.extend([f1, f2])
            seen_features.add(f1); seen_features.add(f2)
    remaining = [f for f in original_feature_names if f not in seen_features]
    new_feature_order.extend(remaining)

    # Indices to reorder columns the same way everywhere
    original_indices = {name: i for i, name in enumerate(original_feature_names)}
    new_indices = [original_indices[name] for name in new_feature_order]

    # Apply the SAME reordering to train/val/test
    X_train = X_train[:, new_indices]
    X_val   = X_val[:,   new_indices]
    X_test  = X_test[:,  new_indices]

    # --- Build the global priority map from TRAIN-ONLY correlations + CHUNK constraints ---
    # Map each feature to (chunk_id, qubit_id) after reordering
    n_qubits = CONFIG["n_qubits"]
    feature_to_qubit_map = {name: (i // n_qubits, i % n_qubits) for i, name in enumerate(new_feature_order)}

    priority_entanglement_map.clear()  # reset for this run
    deg_cap = 50                        # max entanglement degree per feature per chunk
    deg = {f: 0 for f in new_feature_order}
    top_pairs_found = 0

    for (f1, f2), _ in pairs_series.items():
        if top_pairs_found >= CONFIG["num_priority_pairs"]:
            break
        c1, q1 = feature_to_qubit_map[f1]
        c2, q2 = feature_to_qubit_map[f2]
        # only entangle if both features land in the SAME re-upload chunk and degree cap allows
        if c1 == c2 and deg[f1] < deg_cap and deg[f2] < deg_cap:
            priority_entanglement_map.append({'chunk': c1, 'qubits': [q1, q2], 'features': {f1, f2}})
            deg[f1] += 1; deg[f2] += 1
            top_pairs_found += 1

    print(f"Found {len(priority_entanglement_map)} viable priority pairs to entangle (train-only, degree-capped).")

    # --- Print & Save: Priority entanglement pairs by chunk ---
    if len(priority_entanglement_map):
        rows = []
        for rank, item in enumerate(priority_entanglement_map, start=1):
            # item['features'] is a set; sort for deterministic printing
            f1, f2 = sorted(list(item['features']))
            q0, q1 = item['qubits']
            rows.append({
                "rank": rank,
                "chunk": item["chunk"],
                "feature_1": f1,
                "feature_2": f2,
                "qubit_1": q0,
                "qubit_2": q1,
            })
        df_pairs = pd.DataFrame(rows).sort_values(["chunk", "rank"])
        print("\n--- Priority entanglement pairs (by chunk) ---")
        print(df_pairs.to_string(index=False))
        csv_path = output_dir / "priority_entanglement_pairs.csv"
        df_pairs.to_csv(csv_path, index=False)
        print(f"Saved entanglement pairs to {csv_path}")
    else:
        print("No priority entanglement pairs selected.")
    
#     # --- Render the circuit diagram (uses current CONFIG and priority_entanglement_map) ---
#     render_circuit_diagrams(output_dir, n_features=X.shape[1], config=CONFIG)

    # --- Scaling and PyTorch DataLoaders ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    # guard against divide-by-zero if no positives (unlikely but safe)
    n_pos = max(1, int((y_train_tensor == 1).sum().item()))
    n_neg = int((y_train_tensor == 0).sum().item())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train_tensor), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val,   dtype=torch.float32), torch.tensor(y_val,  dtype=torch.float32)), batch_size=32)
    test_loader  = DataLoader(TensorDataset(torch.tensor(X_test,  dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=32)

    # --- Save the essential preprocessing artifacts for this run ---
    with open(output_dir / 'scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    with open(output_dir / 'feature_order.pkl', 'wb') as f: pickle.dump(new_indices, f)
    print(f"Scaler and Feature Order saved to {output_dir}")

    # --- Run Experiments for both models ---
    n_features = X_train.shape[1]; results = {}
    
    # Train, evaluate, and save the Hybrid model
    hybrid_model, hybrid_results = run_experiment('Hybrid (Custom)', n_features, CONFIG, qnode=custom_quantum_circuit, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, pos_weight=pos_weight)
    torch.save(hybrid_model.state_dict(), output_dir / "Hybrid_(Custom)_weights.pth"); results['Hybrid (Custom)'] = hybrid_results
    
    # Train, evaluate, and save the Classical model
    classical_model, classical_results = run_experiment('Classical', n_features, CONFIG, qnode=None, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, pos_weight=pos_weight)
    torch.save(classical_model.state_dict(), output_dir / "Classical_weights.pth"); results['Classical'] = classical_results
    
    # --- Log, Present, and Display Final Results ---
    log_results_to_file("0p-10q-1l.txt", run_timestamp, CONFIG, results)
    present_results(CONFIG, results)
    display_model_structures(hybrid_model, classical_model, n_features)

# # =============================================================================
# # SECTION 6A: CIRCUIT VISUALIZATION
# # =============================================================================
# def render_circuit_diagrams(output_dir: Path, n_features: int, config: dict):
#     """
#     Renders the quantum circuit as ASCII in the console and saves a PNG.
#     Uses the current global priority_entanglement_map and CONFIG.
#     """
#     n_qubits = config["n_qubits"]
#     num_chunks = int(np.ceil(n_features / n_qubits))

#     # Dummy inputs/weights just to render the topology (values don't matter)
#     dummy_inputs = torch.zeros(n_features, dtype=torch.float32)
#     dummy_weights = torch.zeros((num_chunks, config["num_q_layers"], n_qubits, 3), dtype=torch.float32)

#     # ASCII diagram
#     print("\n--- Quantum circuit (ASCII) ---")
#     ascii_diagram = qml.draw(custom_quantum_circuit)(dummy_inputs, dummy_weights)
#     print(ascii_diagram)
#     with open(output_dir / "circuit.txt", "w") as f:
#         f.write(ascii_diagram)
#     print(f"Saved ASCII circuit to {output_dir / 'circuit.txt'}")

#     # Matplotlib diagram (if available)
#     try:
#         fig = qml.draw_mpl(custom_quantum_circuit)(dummy_inputs, dummy_weights)
#         fig.suptitle(f"Quantum Circuit | qubits={n_qubits}, chunks={num_chunks}", fontsize=10)
#         png_path = output_dir / "circuit.png"
#         fig.savefig(png_path, dpi=300, bbox_inches="tight")
#         plt.close(fig)
#         print(f"Saved circuit diagram to {png_path}")
#     except Exception as e:
#         print(f"Matplotlib circuit drawing not available: {e}")
    
    
# =============================================================================
# SECTION 7: TRAINING AND EVALUATION FUNCTION
# =============================================================================
def run_experiment(model_type, n_features, config, qnode, train_loader, val_loader, test_loader, pos_weight):
    """Initializes, trains with per-epoch monitoring, and evaluates a single model."""
    print(f"\n----- Running Experiment for: {model_type} Model -----")
    model = HybridModel(n_features=n_features, config=config, qnode=qnode)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    start_time = time.time()
    for epoch in range(config["epochs"]):
        # --- Training Phase for the current epoch ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", leave=False)
        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0).squeeze().long()
            train_total += labels.size(0)
            train_correct += (predicted == labels.long()).sum().item()
            progress_bar.set_postfix(loss=f"{train_loss/train_total:.4f}")

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100. * train_correct / train_total

        # --- Validation Phase to monitor generalization ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad(): # Disable gradient calculation for efficiency
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0).squeeze().long()
                val_total += labels.size(0)
                val_correct += (predicted == labels.long()).sum().item()
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total

        # --- Print Per-Epoch Summary ---
        print(
            f"Epoch {epoch+1:02d}/{config['epochs']} | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%"
        )

    print(f"Training finished in {time.time() - start_time:.2f} seconds.")
    
    # --- Final Evaluation on the held-out Test Set ---
    model.eval(); y_true, y_pred_proba = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs); probs = torch.sigmoid(outputs); y_true.extend(labels.tolist()); y_pred_proba.extend(probs.squeeze().tolist())
    y_pred = (np.array(y_pred_proba) > config["threshold"]).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    eval_results = {'Accuracy': accuracy_score(y_true, y_pred), 'Precision': precision_score(y_true, y_pred, zero_division=0), 'Recall': recall_score(y_true, y_pred, zero_division=0), 'F1-Score': f1_score(y_true, y_pred, zero_division=0), 'ROC-AUC': roc_auc_score(y_true, y_pred_proba), 'PR-AUC': pr_auc, 'Confusion Matrix': confusion_matrix(y_true, y_pred), 'y_true': y_true, 'y_pred_proba': y_pred_proba}
    return model, eval_results

# =============================================================================
# SECTION 8: UTILITY FUNCTIONS FOR LOGGING AND PRESENTATION
# =============================================================================
def log_results_to_file(filename, timestamp, config, results):
    """Appends the key results of a run to a master CSV log file."""
    log_path = Path(filename)
    # Create file and write header if it doesn't exist
    if not log_path.exists():
        with open(log_path, "w") as f:
            header = "timestamp," + ",".join(config.keys()) + ",model_type,precision,recall,pr_auc\n"
            f.write(header)
    
    # Append the results for each model in this run
    with open(log_path, "a") as f:
        config_str = ",".join(map(str, config.values()))
        for model_name, result_data in results.items():
            precision, recall, pr_auc = result_data['Precision'], result_data['Recall'], result_data['PR-AUC']
            f.write(f"{timestamp},{config_str},{model_name},{precision:.4f},{recall:.4f},{pr_auc:.4f}\n")
    print(f"\nResults successfully logged to {filename}")

def present_results(config, results):
    """Prints and plots the final comparisons for the current run."""
    # Create a dynamic title suffix based on the current configuration.
    test_split = 1.0 - config['train_split'] - config['val_split']
    title_suffix = (
        f"(Sample: {config['sample_size']}, Split: {int(config['train_split']*100)}/{int(config['val_split']*100)}/{int(test_split*100)}, "
        f"Qubits: {config['n_qubits']}, Q-Layers: {config['num_q_layers']}, Pairs: {config['num_priority_pairs']})"
    )
    
    # --- Print Final Metrics Table ---
    print("\n\n" + "="*85); print("--- COMPARISON OF FINAL METRICS ---"); print("="*85)
    print(f"Configuration: {title_suffix}")
    print(f"{'Metric':<20} | {'Hybrid (Custom)':<20} | {'Classical Model':<20}"); print("-" * 85)
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']:
        cust_val, clas_val = f"{results['Hybrid (Custom)'][metric]:.4f}", f"{results['Classical'][metric]:.4f}"
        print(f"{metric:<20} | {cust_val:<20} | {clas_val:<20}")
    
    # --- Print Confusion Matrices ---
    print("\n" + "-"*85); print("--- Confusion Matrices ---"); print("-" * 85)
    for model_name, result_data in results.items():
        print(f"\n{model_name} Confusion Matrix:"); cm = result_data['Confusion Matrix']; print(cm)
        tn, fp, fn, tp = cm.ravel(); print(f"True Negatives: {tn}, False Positives: {fp}\nFalse Negatives: {fn}, True Positives: {tp}")
    
    # --- Generate Visualizations ---
    print("\n" + "="*85); print("--- VISUALIZATIONS ---"); print("="*85)
    
    # Plot 1: Precision-Recall Curve
    plt.figure(figsize=(10, 7)); plt.title(f'Precision-Recall Curve\n{title_suffix}', fontsize=14)
    for model_name, result_data in results.items():
        precision, recall, _ = precision_recall_curve(result_data['y_true'], result_data['y_pred_proba']); plt.plot(recall, precision, lw=2, label=f"{model_name} (PR-AUC = {result_data['PR-AUC']:.4f})")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(loc='best'); plt.grid(True); plt.show()
    
    # Plot 2: Cumulative Gains Chart
    fig, ax = plt.subplots(figsize=(10, 7))
    for model_name, line_style in [('Hybrid (Custom)', '-'), ('Classical', '--')]:
        y_true, y_probas = results[model_name]['y_true'], results[model_name]['y_pred_proba']
        df = pd.DataFrame({'y_true': y_true, 'y_probas': y_probas}).sort_values('y_probas', ascending=False)
        df['gain'] = df['y_true'].cumsum() / df['y_true'].sum(); df['percentage_sample'] = np.arange(1, len(df) + 1) / len(df)
        ax.plot(df['percentage_sample'].values, df['gain'].values, lw=2, linestyle=line_style, label=model_name)
    ax.plot([0, 1], [0, 1], lw=2, linestyle=':', color='black', label='Baseline')
    ax.set_title(f'Cumulative Gains Chart\n{title_suffix}', fontsize=14); ax.set_xlabel('Percentage of Sample'); ax.set_ylabel('Gain'); ax.legend(loc='best'); ax.grid(True); plt.show()

def display_model_structures(hybrid_model, classical_model, n_features):
    """Prints a summary of the model architectures using torchinfo."""
    if torchinfo is None:
        print("\nSkipping model summary display because torchinfo is not installed.")
        return

    print("\n\n" + "="*85); print("--- MODEL ARCHITECTURES ---"); print("="*85)
    input_size = (1, n_features)

    print("\n--- Hybrid (Custom) Model Structure ---")
    summary_hybrid = torchinfo.summary(hybrid_model, input_size=input_size, depth=5, col_names=["input_size", "output_size", "num_params", "mult_adds"])
    print(summary_hybrid)

    print("\n--- Classical Model Structure ---")
    summary_classical = torchinfo.summary(classical_model, input_size=input_size, depth=5, col_names=["input_size", "output_size", "num_params", "mult_adds"])
    print(summary_classical)

# =============================================================================
# SECTION 9: SCRIPT ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    # Execute the main experimental function.
    main()
