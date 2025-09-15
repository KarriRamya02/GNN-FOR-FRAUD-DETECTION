import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

class SimpleGNNLayer(nn.Module):
    """Simple Graph Neural Network Layer"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_self = nn.Linear(in_features, out_features)
        self.linear_neighbor = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj_matrix):
        # Self transformation
        self_features = self.linear_self(x)
        
        # Neighbor aggregation
        neighbor_features = torch.mm(adj_matrix, x)
        neighbor_features = self.linear_neighbor(neighbor_features)
        
        # Combine self and neighbor features
        out = self_features + neighbor_features
        return out

class MinimalGNNFraudDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.5):
        super().__init__()
        
        # Graph layers
        self.gnn1 = SimpleGNNLayer(input_dim, hidden_dim)
        self.gnn2 = SimpleGNNLayer(hidden_dim, hidden_dim)
        self.gnn3 = SimpleGNNLayer(hidden_dim, hidden_dim // 2)
        
        # Node-level classification layers (not graph-level)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, adj_matrix):
        # Graph convolutions
        x1 = F.relu(self.gnn1(x, adj_matrix))
        x1 = self.dropout(x1)
        
        x2 = F.relu(self.gnn2(x1, adj_matrix))
        x2 = self.dropout(x2)
        
        x3 = F.relu(self.gnn3(x2, adj_matrix))
        x3 = self.dropout(x3)
        
        # Node-level classification (no pooling)
        node_outputs = self.classifier(x3)
        
        return node_outputs, x3

class MinimalFraudDetectionSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_columns = []
        self.adj_matrix = None
        
    def preprocess_data(self, df):
        """Preprocess the input data"""
        print("Preprocessing data...")
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Identify feature columns
        target_cols = ['is_fraud', 'fraud', 'label', 'target', 'class']
        id_cols = ['id', 'user_id', 'transaction_id', 'account_id']
        
        self.feature_columns = [col for col in df.columns 
                               if col.lower() not in target_cols + id_cols 
                               and df[col].dtype in ['int64', 'float64']]
        
        # Encode categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col.lower() not in target_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                if col not in self.feature_columns:
                    self.feature_columns.append(col)
        
        # Prepare target variable
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            print("No fraud label found, creating synthetic labels...")
            df['is_fraud'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
            target_col = 'is_fraud'
        
        X = df[self.feature_columns].values
        y = df[target_col].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, df
    
    def create_adjacency_matrix(self, X, threshold=0.5):
        """Create adjacency matrix from feature similarity"""
        print("Creating adjacency matrix...")
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        # For large datasets, use a sample to compute similarity
        if len(X) > 1000:
            sample_indices = np.random.choice(len(X), size=1000, replace=False)
            X_sample = X[sample_indices]
            similarity_matrix = cosine_similarity(X_sample)
            
            # Create full adjacency matrix
            adj_matrix = np.zeros((len(X), len(X)))
            for i, idx_i in enumerate(sample_indices):
                for j, idx_j in enumerate(sample_indices):
                    adj_matrix[idx_i, idx_j] = similarity_matrix[i, j]
        else:
            similarity_matrix = cosine_similarity(X)
            adj_matrix = similarity_matrix.copy()
        
        # Create binary adjacency matrix
        adj_matrix = (adj_matrix > threshold).astype(float)
        
        # Add self-connections
        np.fill_diagonal(adj_matrix, 1.0)
        
        # Normalize adjacency matrix (row normalization)
        row_sums = adj_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        adj_matrix = adj_matrix / row_sums[:, np.newaxis]
        
        self.adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)
        
        return self.adj_matrix
    
    def train_model(self, X, y, epochs=50, lr=0.01):
        """Train the GNN model"""
        print("Training GNN model...")
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Initialize model
        input_dim = X.shape[1]
        self.model = MinimalGNNFraudDetector(input_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        # Move data to device
        X_tensor = X_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)
        adj_matrix = self.adj_matrix.to(self.device)
        
        # Training loop
        self.model.train()
        train_losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            node_outputs, embeddings = self.model(X_tensor, adj_matrix)
            
            # Calculate loss for all nodes
            loss = criterion(node_outputs, y_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
        
        return train_losses
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        adj_matrix = self.adj_matrix.to(self.device)
        
        with torch.no_grad():
            node_outputs, embeddings = self.model(X_tensor, adj_matrix)
            probabilities = F.softmax(node_outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy(), embeddings.cpu().numpy()
    
    def create_visualization_data(self, X, predictions, probabilities):
        """Create data for visualization"""
        print("Creating visualization data...")
        
        # For visualization, limit the number of nodes
        max_nodes = min(100, len(X))
        indices = np.random.choice(len(X), size=max_nodes, replace=False) if len(X) > max_nodes else range(len(X))
        
        # Create NetworkX graph from adjacency matrix subset
        adj_subset = self.adj_matrix[indices][:, indices].numpy()
        
        # Remove very weak connections for cleaner visualization
        adj_subset[adj_subset < 0.1] = 0
        
        G = nx.from_numpy_array(adj_subset)
        
        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Prepare visualization data
        nodes_data = []
        edges_data = []
        
        # Create nodes
        for node_id in G.nodes():
            if node_id < len(indices):
                original_idx = indices[node_id]
                if original_idx < len(predictions):
                    x, y = pos[node_id] if node_id in pos else (0, 0)
                    
                    # Get fraud probability (class 1)
                    fraud_prob = probabilities[original_idx][1] if len(probabilities[original_idx]) > 1 else 0.5
                    
                    nodes_data.append({
                        'id': node_id,
                        'x': x * 300 + 400,  # Scale and center
                        'y': y * 300 + 300,
                        'color': 'red' if predictions[original_idx] == 1 else 'green',
                        'fraud_probability': float(fraud_prob),
                        'is_fraud': int(predictions[original_idx])
                    })
        
        # Create edges
        for edge in G.edges():
            if edge[0] < len(nodes_data) and edge[1] < len(nodes_data):
                edges_data.append({
                    'source': edge[0],
                    'target': edge[1]
                })
        
        # Calculate statistics
        fraud_count = sum(1 for n in nodes_data if n['is_fraud'] == 1)
        safe_count = len(nodes_data) - fraud_count
        
        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'statistics': {
                'total_nodes': len(nodes_data),
                'fraud_nodes': fraud_count,
                'safe_nodes': safe_count,
                'total_edges': len(edges_data)
            }
        }

def process_fraud_detection(file_path, save_model_path=None):
    """Main processing function"""
    
    # Initialize system
    fraud_detector = MinimalFraudDetectionSystem()
    
    # Load data
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")
    
    # Preprocess data
    X, y, processed_df = fraud_detector.preprocess_data(df)
    
    # Create adjacency matrix
    adj_matrix = fraud_detector.create_adjacency_matrix(X, threshold=0.5)
    
    # Train model
    train_losses = fraud_detector.train_model(X, y, epochs=50)
    
    # Make predictions
    predictions, probabilities, embeddings = fraud_detector.predict(X)
    
    # Create visualization data
    viz_data = fraud_detector.create_visualization_data(X, predictions, probabilities)
    
    # Print results
    print("\n=== FRAUD DETECTION RESULTS ===")
    print(f"Total transactions: {len(predictions)}")
    print(f"Predicted fraud cases: {sum(predictions)}")
    print(f"Predicted safe cases: {len(predictions) - sum(predictions)}")
    print(f"Fraud rate: {sum(predictions)/len(predictions)*100:.2f}%")
    
    return viz_data, fraud_detector

if __name__ == "__main__":
    sample_file = "sample_data.csv"
    if os.path.exists(sample_file):
        viz_data, model = process_fraud_detection(sample_file)
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        with open("results/graph_data.json", "w") as f:
            json.dump(viz_data, f, indent=2)
        
        print("Results saved!")
    else:
        print(f"Sample file {sample_file} not found!")