#!/bin/bash
set -o xtrace

# Resolution
N_label=n50

# Path to initial conditions
IC_PATH="$HOME/Desktop/swift-cmgi/Demo6/SpinImpactInitCond/demo_impact_${N_label}.hdf5"

# Copy or download the initial conditions if they are not present
if [ ! -e "cmgi_impact_${N_label}.hdf5" ]
then
    if [ -e "$IC_PATH" ]
    then
        cp "$IC_PATH" ./
    else
        ./getICs.sh
    fi
else
    echo "Initial conditions already present: cmgi_impact_${N_label}.hdf5"
fi

# Download equation of state tables if not already present
if [ ! -e "../EoSTables/ANEOS_forsterite_S19.txt" ]
then
    cd ../../EoSTables
    ./get_eos_tables.sh
    cd -
fi

# SWIFT executable path
SWIFT_PATH="$HOME/swiftsim/swift"

# Check if SWIFT exists
if [ ! -f "$SWIFT_PATH" ]; then
    exit 1
fi

# Run SWIFT
"$SWIFT_PATH" --hydro --self-gravity --threads=28 "cmgi_impact_${N_label}.yml" 2>&1 | tee "output_${N_label}.txt"

# Plot the snapshots
python3 plot_snapshots.py