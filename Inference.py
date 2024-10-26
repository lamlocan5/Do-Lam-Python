import streamlit as st
import os
import sys
import traceback
from pathlib import Path
import networkx as nx
import torch
import dgl
from androguard.misc import AnalyzeAPK
from transformers import AutoTokenizer, AutoModel
import pickle
import numpy as np

# Cấu hình thiết bị (GPU nếu có)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tải mô hình CodeT5+ (codet5p-110m-embedding)
@st.cache_resource
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-110m-embedding")
    model_codet5p = AutoModel.from_pretrained("Salesforce/codet5p-110m-embedding").to(device)
    model_codet5p.eval()
    return tokenizer, model_codet5p

tokenizer, model_codet5p = load_embedding_model()

def encode_code_snippet(code_snippet):
    """Tạo embedding cho đoạn mã nguồn sử dụng CodeT5+."""
    inputs = tokenizer(code_snippet, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model_codet5p(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.cpu()

def process_apk(file_data):
    """Xử lý file APK để tạo đồ thị call graph và trích xuất đặc trưng."""
    try:
        with open("temp.apk", "wb") as f:
            f.write(file_data.getbuffer())
        a, d, dxg = AnalyzeAPK("temp.apk")
        cg = dxg.get_call_graph()
        G = nx.DiGraph()
        G.add_edges_from(cg.edges())

        node_embeddings = {}

        for node in G.nodes():
            try:
                if hasattr(node, 'method'):
                    source_code = node.method.get_source()
                    if source_code:
                        embedding = encode_code_snippet(source_code)
                    else:
                        embedding = torch.zeros(768)
                else:
                    embedding = torch.zeros(768)
                node_embeddings[node] = embedding
            except Exception as e:
                node_embeddings[node] = torch.zeros(768)

        nx.set_node_attributes(G, {node: {'feature': node_embeddings[node]} for node in G.nodes()})
        G = nx.convert_node_labels_to_integers(G)

        # Chuyển đổi NetworkX thành DGLGraph
        dg = dgl.from_networkx(G, node_attrs=['feature'])

        # Xóa file tạm
        os.remove("temp.apk")

        return dg

    except Exception as e:
        print(f"Error while processing APK: {e}")
        traceback.print_exception(*sys.exc_info())
        return None

class GCN(torch.nn.Module):
    def __init__(self, in_feats=768, hidden_dims=[256, 128, 64], num_classes=2):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        # Input layer
        self.layers.append(dgl.nn.GraphConv(in_feats, hidden_dims[0]))
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(dgl.nn.GraphConv(hidden_dims[i], hidden_dims[i+1]))
        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, g):
        h = g.ndata['feature'].float().to(device)
        for layer in self.layers:
            h = torch.relu(layer(g.to(device), h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        out = self.fc(hg)
        return out

    def extract_features(self, g):
        """Trích xuất đặc trưng từ đồ thị."""
        self.eval()
        with torch.no_grad():
            h = g.ndata['feature'].float().to(device)
            for layer in self.layers:
                h = torch.relu(layer(g.to(device), h))
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
        return hg.cpu()

@st.cache_resource
def load_models(gcn_model_path, rf_model_path):
    """Load mô hình GCN và Random Forest đã được huấn luyện."""
    # Load GCN model
    gcn_model = GCN().to(device)
    gcn_model.load_state_dict(torch.load(gcn_model_path, map_location=device))
    gcn_model.eval()

    # Load Random Forest model
    with open(rf_model_path, 'rb') as f:
        rf_model = pickle.load(f)

    return gcn_model, rf_model

def main():
    st.title("Malware Detection for APK Files")
    st.write("Upload an APK file to detect whether it's malicious or benign.")

    uploaded_file = st.file_uploader("Choose an APK file", type=["apk"])

    if uploaded_file is not None:
        st.write("Processing the APK file...")
        graph = process_apk(uploaded_file)
        if graph is not None:
            st.write("Extracting features using GCN...")
            # Load models
            gcn_model_path = "gcn_model.pt"  # Thay thế bằng đường dẫn tới mô hình GCN của bạn
            rf_model_path = "gcn_model_rf.pkl"  # Thay thế bằng đường dẫn tới mô hình Random Forest của bạn
            gcn_model, rf_model = load_models(gcn_model_path, rf_model_path)

            # Extract features
            features = gcn_model.extract_features(graph).numpy().reshape(1, -1)

            st.write("Predicting using Random Forest...")
            prediction = rf_model.predict(features)
            probability = rf_model.predict_proba(features)

            if prediction[0] == 1:
                st.error(f"The APK file is **Malware** with probability {probability[0][1]*100:.2f}%")
            else:
                st.success(f"The APK file is **Benign** with probability {probability[0][0]*100:.2f}%")
        else:
            st.error("Failed to process the APK file. Please ensure it's a valid APK.")

if __name__ == "__main__":
    main()
