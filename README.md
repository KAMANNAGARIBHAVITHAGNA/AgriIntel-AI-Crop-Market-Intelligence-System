# AgriIntel-AI-Crop-Market-Intelligence-System
AgriIntel AI An AI-powered agricultural platform that detects crop diseases from images and predicts market prices using EfficientNet and XGBoost models. Features a conversational RAG interface and interactive 3D visualization for real-time crop health monitoring and market intelligence. Built with FastAPI, React, and PyTorch.
🔬 1. Computer Vision Module (Disease Detection)
Theory
Model: EfficientNet-B0 (Convolutional Neural Network)

Why EfficientNet?

Compound scaling method balances depth, width, and resolution
Efficient architecture with fewer parameters than ResNet
Pre-trained on ImageNet, fine-tuned on crop disease dataset
How It Works:

Input: Crop image (224x224 RGB)
Preprocessing: Normalization using ImageNet statistics
Feature Extraction: EfficientNet backbone extracts visual features
Classification: Fully connected layer outputs disease probabilities
Output: Disease label (Healthy/Powdery/Rust) + confidence score
Training Process:

Dataset: 3 classes (Healthy, Powdery Mildew, Rust Disease)
Loss Function: Cross-Entropy Loss
Optimizer: Adam with learning rate scheduling
Data Augmentation: Random flips, rotations, color jitter
Epochs: 5 (achieves 81% test accuracy)
Mathematical Foundation:

P(disease|image) = softmax(EfficientNet(image))
Risk Level = f(confidence, disease_type)
📊 2. Price Prediction Module (Market Forecasting)
Theory
Model: XGBoost (Gradient Boosted Decision Trees)

Why XGBoost?

Handles non-linear relationships in market data
Robust to missing values and outliers
Fast training with parallel processing
Built-in regularization prevents overfitting
Feature Engineering:

Temporal Features:

Day of month, Month, Day of week
Captures seasonal patterns
Lag Features:

lag_1: Previous day's price
lag_7: Price from 7 days ago
Captures price momentum and trends
Categorical Features:

State, District, Market, Commodity, Variety, Grade
Label encoded for tree-based models
How It Works:

Input: Crop name, location, temporal context
Feature Construction: Build feature vector with lags + encodings
Prediction: XGBoost ensemble predicts price
Trend Analysis: Compare with historical averages
Output: Predicted price + trend direction + confidence
Mathematical Foundation:

Price(t) = XGBoost(State, District, Commodity, lag_1, lag_7, day, month, dayofweek)
Trend = (Price(t) - Price(t-7)) / Price(t-7) * 100
Training Process:

Dataset: Historical daily_price.csv (multiple crops, regions, dates)
Objective: Regression (minimize RMSE)
Hyperparameters: max_depth=6, learning_rate=0.1, n_estimators=100
Validation: Train/test split with temporal ordering preserved
🧠 3. RAG Module (Retrieval-Augmented Generation)
Theory
Components:

Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)
Vector Database: FAISS (Facebook AI Similarity Search)
Retrieval: Semantic similarity search
Why RAG?

Provides contextual explanations grounded in real market data
Reduces hallucinations by retrieving factual information
Scalable to large knowledge bases
How It Works:

Indexing Phase (Offline):

Collect market insights, agricultural knowledge
Generate embeddings using Sentence-Transformers
Build FAISS index for fast similarity search
Query Phase (Online):

User query → Embedding vector
FAISS retrieves top-k similar documents
Context + Query → Generate response
Output: Contextual explanation with retrieved facts

Mathematical Foundation:

query_embedding = SentenceTransformer(query)
similar_docs = FAISS.search(query_embedding, k=2)
response = generate(query, context=similar_docs)
🎮 4. Intent Detection & Routing
Theory
Rule-Based Intent Classification:

if image_uploaded:
    intent = "VISION"  # Disease detection
elif "price" in query or "market" in query:
    intent = "MARKET"  # Price prediction
elif "what disease" in query:
    intent = "INFO"  # System capabilities
else:
    intent = "MARKET"  # Default
Why This Approach?

Fast and deterministic
No additional model training required
Easy to debug and extend
Sufficient for domain-specific application
🏗️ System Architecture
Data Flow
User Query/Image
      ↓
[Intent Detection]
      ↓
   ┌──┴──┐
   ↓     ↓
[VISION] [MARKET]
   ↓     ↓
[EfficientNet] [XGBoost]
   ↓     ↓
   └──┬──┘
      ↓
  [RAG Context]
      ↓
[Response Generation]
      ↓
[3D Visualization Update]
Backend (FastAPI)
Async Processing: Non-blocking I/O for concurrent requests
Model Loading: On startup, models loaded into memory
CORS Enabled: Cross-origin requests from frontend
RESTful API: JSON request/response format
Frontend (React + Three.js)
State Management: React Context API for global state
3D Rendering: React Three Fiber (declarative Three.js)
Animations: Framer Motion for smooth transitions
Camera Control: Dynamic positioning based on mode (Satellite/Drone)
📐 Mathematical Models Summary
1. Disease Detection
y = argmax(softmax(W · EfficientNet(x) + b))
where:
  x = input image (224×224×3)
  y = predicted class {0: Healthy, 1: Powdery, 2: Rust}
2. Price Prediction
Price = Σ(tree_i(features))  for i in [1..100]
where:
  features = [State_enc, District_enc, Commodity_enc, lag_1, lag_7, day, month, dow]
  tree_i = individual decision tree in XGBoost ensemble
3. RAG Retrieval
similarity(q, d) = cosine(embed(q), embed(d))
retrieved_docs = top_k(similarity(query, corpus))
🔄 Training Pipeline
Complete Workflow
1. Data Preparation
   ├── Load daily_price.csv
   ├── Load crop images (Train/Test split)
   └── Prepare RAG corpus

2. Model Training
   ├── train_xgb.py → price_model.pkl
   ├── train_cv.py → cv_model.pth
   └── build_rag.py → faiss_index.bin

3. Model Deployment
   ├── FastAPI loads models on startup
   └── Models serve predictions via REST API

4. Frontend Integration
   └── React app consumes API endpoints
🎯 Key Innovations
Multi-Modal Integration: Seamlessly combines vision, tabular, and text modalities
Intent-Based Routing: Automatically selects appropriate model based on query
Real-Time 3D Visualization: Interactive camera transitions enhance UX
RAG-Enhanced Explanations: Provides factual, grounded market insights
Production-Ready: Complete training pipeline with error handling
📊 Performance Characteristics
Model Metrics
EfficientNet: 81% accuracy on test set (3-class classification)
XGBoost: R² > 0.80 on price prediction (regression)
FAISS: Sub-millisecond retrieval for 1000+ documents
System Performance
API Latency: <500ms for disease detection
Price Prediction: <100ms response time
RAG Retrieval: <50ms for top-k search
3D Rendering: 60 FPS on modern hardware
🔬 Technical Stack Justification
Component	Technology	Reason
Backend Framework	FastAPI	Async support, automatic API docs, type safety
ML Framework	PyTorch	Dynamic computation graphs, research-friendly
Tabular ML	XGBoost	Best-in-class for structured data
Vector DB	FAISS	Fastest similarity search, GPU support
Embeddings	Sentence-Transformers	Pre-trained, efficient, multilingual
Frontend	React	Component-based, large ecosystem
3D Engine	Three.js	Industry standard, WebGL support
Animation	Framer Motion	Declarative, smooth transitions
