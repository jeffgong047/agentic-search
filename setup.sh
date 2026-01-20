#!/bin/bash

# Setup script for High-SNR Agentic RAG System

echo "=================================="
echo "High-SNR Agentic RAG - Setup"
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment (optional but recommended)
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env and add your OPENAI_API_KEY"
else
    echo ""
    echo ".env file already exists"
fi

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Set your OpenAI API key in .env or:"
echo "   export OPENAI_API_KEY='your-key-here'"
echo "3. Run the test: python main.py"
echo "4. Run examples: python example.py"
echo ""
echo "Optional - Start Elasticsearch (for ES integration):"
echo "docker run -d -p 9200:9200 -e \"discovery.type=single-node\" elasticsearch:8.0.0"
echo ""
