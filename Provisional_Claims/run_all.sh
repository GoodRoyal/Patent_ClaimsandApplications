#!/bin/bash
# Run all patent validations
# Usage: ./run_all.sh

echo "=============================================="
echo "ENTROPY SYSTEMS - PATENT VALIDATION SUITE"
echo "=============================================="
echo ""

# Check Python
python3 --version || { echo "Python 3 required"; exit 1; }

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Running Patent 1a: IT-OFNG..."
echo "----------------------------------------------"
python3 Patent_1a/patent_1a_real.py

echo ""
echo "Running Patent 2a: Substrate Orchestration..."
echo "----------------------------------------------"
python3 Patent_2a/patent_2a_real.py

echo ""
echo "Running Patent 3a: Thermodynamic Phase Inference..."
echo "----------------------------------------------"
python3 Patent_3a/patent_3a_real.py

echo ""
echo "Running Patent 4a: CSOS..."
echo "----------------------------------------------"
python3 Patent_4a/patent_4a_real.py

echo ""
echo "Running Patent 5a: Sheaf Sensor Fusion..."
echo "----------------------------------------------"
python3 Patent_5a/patent_5a_real.py

echo ""
echo "=============================================="
echo "ALL VALIDATIONS COMPLETE"
echo "=============================================="
