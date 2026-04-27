# Create and activate venv
echo "Creating virtual environment..."
uv venv .venv
source .venv/Scripts/activate
echo "Virtual environment created and activated."

# Install from requirements.txt
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt
echo "Dependencies installed successfully."