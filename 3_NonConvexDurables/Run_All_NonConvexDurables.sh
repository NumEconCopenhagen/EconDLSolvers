#!/bin/bash

# Script to run all paper result notebooks in sequential order on Ucloud
# Excludes 01b_Run_DP.ipynb (requires C++ and Windows)

# - 3_NonConvexDurables: ~21.2 hours (1270 minutes)

set -e  # Exit on any error

echo "Starting paper results pipeline..."
echo ""

# 3_NonConvexDurables (~21.2 hours)
echo "Running 3_NonConvexDurables notebooks..."

echo "1/8: Running 02a_Run_DL.ipynb (est. 600 min = 10 hours)..."
jupyter nbconvert --to notebook --execute --inplace 3_NonConvexDurables/02a_Run_DL.ipynb --ExecutePreprocessor.timeout=-1

echo "2/8: Running 02b_Backward_Improvement.ipynb (est. 240 min = 4 hours)..."
jupyter nbconvert --to notebook --execute --inplace 3_NonConvexDurables/02b_Backward_Improvement.ipynb --ExecutePreprocessor.timeout=-1

echo "3/8: Running 02c_Run_DL_extra.ipynb (est. 360 min = 6 hours)..."
jupyter nbconvert --to notebook --execute --inplace 3_NonConvexDurables/02c_Run_DL_extra.ipynb --ExecutePreprocessor.timeout=-1

echo "4/8: Running 02d_Run_DL_par_change.ipynb (est. 60 min = 1 hour)..."
jupyter nbconvert --to notebook --execute --inplace 3_NonConvexDurables/02d_Run_DL_par_change.ipynb --ExecutePreprocessor.timeout=-1

echo "5/8: Running 03_Results_DHR26.ipynb (est. 1 min)..."
jupyter nbconvert --to notebook --execute --inplace 3_NonConvexDurables/03_Results_DHR26.ipynb --ExecutePreprocessor.timeout=-1
