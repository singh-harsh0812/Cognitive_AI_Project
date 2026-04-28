# Cognitive_AI_Project
Cognitive Decision Support System using NLP, Logistic Regression, FLAN-T5, Simulation Engine, and Streamlit UI

Cognitive AI Project
Python Version Required

This project uses Python 3.10 because some libraries are not stable on Python 3.14.

Step 1: Install Python 3.10
winget install Python.Python.3.10

Verify installation:

py -3.10 --version
Step 2: Create Virtual Environment
py -3.10 -m venv venv310

Activate it:

.\venv310\Scripts\Activate

Check Python version:

python --version

It should show Python 3.10.x

Step 3: Install Required Libraries
pip install transformers peft accelerate sentencepiece streamlit pandas scikit-learn joblib textblob

Install PyTorch with CUDA:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
Step 4: Verify GPU Support
python -c "import torch; print(torch.cuda.is_available())"

It should return:

True
Step 5: when want to Deactivate Virtual Environment
deactivate