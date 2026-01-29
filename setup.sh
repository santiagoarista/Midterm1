#!/bin/bash

# Quickstart script for Explainable Credit Risk Project
# Run this script to set up the environment and run the full pipeline

set -e

echo "=============================================="
echo "Explainable Credit Risk - Quick Setup"
echo "=============================================="

# Check Python version
echo ""
echo "[1/5] Checking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "[2/5] Creating virtual environment..."
    python3 -m venv venv
else
    echo ""
    echo "[2/5] Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[3/5] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "[4/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if data exists
echo ""
echo "[5/5] Checking for dataset..."
if [ ! -f "data/application_train.csv" ]; then
    echo ""
    echo "⚠️  WARNING: Dataset not found!"
    echo ""
    echo "Please download the Home Credit Default Risk dataset from:"
    echo "https://www.kaggle.com/competitions/home-credit-default-risk/data"
    echo ""
    echo "Then place the CSV files in the 'data/' directory:"
    echo "  - data/application_train.csv"
    echo "  - data/application_test.csv"
    echo ""
    echo "After downloading the data, run:"
    echo "  python src/train.py"
else
    echo "✓ Dataset found!"
    echo ""
    echo "=============================================="
    echo "Setup Complete!"
    echo "=============================================="
    echo ""
    echo "To run the full pipeline:"
    echo "  python src/train.py"
    echo ""
    echo "To compile the paper:"
    echo "  cd paper && pdflatex midterm1.tex"
fi
