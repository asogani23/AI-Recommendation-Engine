# AI Recommendation Engine

## Overview
This project implements a **collaborative filtering-based recommendation engine** using **PyTorch** for the model and **Flask** for serving the API. The application demonstrates end-to-end machine learning development, including model training, deployment, and serving recommendations via RESTful APIs.

---

## Key Features
1. **PyTorch-Based Recommendation Model**:
   - Implements a collaborative filtering approach to generate personalized recommendations.
   - Model architecture: Single-layer neural network for simplicity.

2. **REST API**:
   - `/recommend`: Accepts user input (`user_id`) and returns recommendations.
   - `/health`: Confirms the server's health and operational status.


3. **Model Deployment**:
   - The trained model is saved as `recommendation_model.pt`.
   - The model is loaded during runtime for real-time inference.

4. **Flask**:
   - Lightweight framework to handle API requests and responses.

---

## Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/asogani23/AI_Recommendation_Engine.git
cd AI_Recommendation_Engine
Set Up Virtual Environment
python3 -m venv venv
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Train the Model (Optional)
If you want to retrain the model, run:
python model.py
This will generate a new recommendation_model.pt file.

Start the Flask App
python app.py
The app will run at http://127.0.0.1:5001.
Example API Usage
Test /recommend Endpoint
Send a POST request with a user_id to get recommendations:
curl -X POST -H "Content-Type: application/json" \
-d '{"user_id": 1}' \
http://127.0.0.1:5001/recommend
Expected Response:
{
  "user_id": 1,
  "recommendations": [
    [0.0017]
  ]
}
Test /health Endpoint
Send a GET request to check the server's health:
curl http://127.0.0.1:5001/health
Expected Response:
{
  "status": "ok"
}



Here’s the README.md for your AI Recommendation Engine and a separate document for explaining and discussing the project during technical interviews.

README.md for AI Recommendation Engine
markdown
Copy code
# AI Recommendation Engine

## Overview
This project implements a **collaborative filtering-based recommendation engine** using **PyTorch** for the model and **Flask** for serving the API. The application demonstrates end-to-end machine learning development, including model training, deployment, and serving recommendations via RESTful APIs.

---

## Key Features
1. **PyTorch-Based Recommendation Model**:
   - Implements a collaborative filtering approach to generate personalized recommendations.
   - Model architecture: Single-layer neural network for simplicity.

2. **REST API**:
   - `/recommend`: Accepts user input (`user_id`) and returns recommendations.
   - `/health`: Confirms the server's health and operational status.

3. **Model Deployment**:
   - The trained model is saved as `recommendation_model.pt`.
   - The model is loaded during runtime for real-time inference.

4. **Flask**:
   - Lightweight framework to handle API requests and responses.

---

## Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/asogani23/AI_Recommendation_Engine.git
cd AI_Recommendation_Engine
Set Up Virtual Environment
bash
Copy code
python3 -m venv venv
source venv/bin/activate
Install Dependencies
bash
Copy code
pip install -r requirements.txt
Train the Model (Optional)
If you want to retrain the model, run:

bash
Copy code
python model.py
This will generate a new recommendation_model.pt file.

Start the Flask App
bash
Copy code
python app.py
The app will run at http://127.0.0.1:5001.

Example API Usage
Test /recommend Endpoint
Send a POST request with a user_id to get recommendations:

bash
Copy code
curl -X POST -H "Content-Type: application/json" \
-d '{"user_id": 1}' \
http://127.0.0.1:5001/recommend
Expected Response:

json
Copy code
{
  "user_id": 1,
  "recommendations": [
    [0.0017]
  ]
}
Test /health Endpoint
Send a GET request to check the server's health:

bash
Copy code
curl http://127.0.0.1:5001/health
Expected Response:

json
Copy code
{
  "status": "ok"
}
Technical Details
Model:

Single-layer neural network trained with dummy data.
Saves the trained weights as recommendation_model.pt.
Flask Application:

Lightweight server for handling API requests.
Runs on http://127.0.0.1:5001.
Dependencies:

Flask: For API development.
PyTorch: For machine learning model training and inference.
Challenges and Solutions
FutureWarning in PyTorch:

Updated torch.load to include weights_only=True for security and future compatibility.
Port Conflict:

Changed the default port to 5001 to resolve conflicts with other applications.
Training Size Mismatch:

Ensured that the model's output size matches the target size during training.














