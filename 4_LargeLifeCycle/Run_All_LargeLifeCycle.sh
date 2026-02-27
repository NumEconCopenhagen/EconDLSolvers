#!/bin/bash

# Script to run all paper result notebooks in sequential order on Ucloud

# - 4_LargeLifeCycle: ~16.2 hours (970 minutes)

set -e  # Exit on any error

echo "Starting paper results pipeline..."
echo ""

# 4_LargeLifeCycle (~16.2 hours)
echo ""
echo "Running 4_LargeLifeCycle notebooks..."

echo "6/8: Running 01_Run.ipynb (est. 720 min = 12 hours)..."
jupyter nbconvert --to notebook --execute --inplace 4_LargeLifeCycle/01_Run.ipynb --ExecutePreprocessor.timeout=-1

echo "7/8: Running 02_Backward_test.ipynb (est. 240 min = 4 hours)..."
jupyter nbconvert --to notebook --execute --inplace 4_LargeLifeCycle/02_Backward_test.ipynb --ExecutePreprocessor.timeout=-1

echo "8/8: Running 03_Results_DHR26.ipynb (est. 1 min)..."
jupyter nbconvert --to notebook --execute --inplace 4_LargeLifeCycle/03_Results_DHR26.ipynb --ExecutePreprocessor.timeout=-1

echo ""
echo "All notebooks completed successfully!"
