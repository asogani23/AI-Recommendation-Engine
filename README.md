## AI Recommendation Engine

### Overview
This project implements a collaborative filtering-based recommendation system using PyTorch. It serves personalized recommendations through a Flask API.

### Features
- Matrix Factorization using PyTorch.
- Flask-based API for serving real-time recommendations.
- Deployed on AWS EC2 with a pre-trained model stored in S3.

### How to Run
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (Optional):
   ```bash
   python model.py
   ```

3. **Start Flask API**:
   ```bash
   python app.py
   ```

4. **Access Endpoints**:
   - `POST /recommend` to get recommendations for a user.
