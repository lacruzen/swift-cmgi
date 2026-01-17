#!/bin/bash
set -o xtrace
set -e  # Exit on any error
set -u  # Exit on undefined variables
set -o pipefail  # Capture pipe failures

# ============================================================================
# CONFIGURATION
# ============================================================================
readonly N_label="n50"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SWIFT_PATH="${SWIFT_PATH:-$HOME/swiftsim/swift}"
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
readonly EOS_TABLES_DIR="$PROJECT_ROOT/EoSTables"
readonly REQUIRED_EOS_FILES=("ANEOS_forsterite_S19.txt" "ANEOS_Fe85Si15_S20.txt")

# ============================================================================
# FUNCTIONS
# ============================================================================
print_header() {
    echo "================================================================"
    echo "$1"
    echo "================================================================"
}

print_success() {
    echo "✅ $1"
}

print_error() {
    echo "❌ ERROR: $1" >&2
}

print_warning() {
    echo "⚠️  WARNING: $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "Command '$1' not found"
        return 1
    fi
}

check_file_exists() {
    if [[ ! -f "$1" ]]; then
        print_error "File not found: $1"
        return 1
    fi
}

check_directory_exists() {
    if [[ ! -d "$1" ]]; then
        print_error "Directory not found: $1"
        return 1
    fi
}

validate_environment() {
    print_header "VALIDATING ENVIRONMENT"
    
    # Check Python
    if ! check_command "python3"; then
        print_error "Python3 is required but not found"
        return 1
    fi
    
    # Check SWIFT executable
    if ! check_file_exists "$SWIFT_PATH"; then
        print_error "SWIFT executable not found at: $SWIFT_PATH"
        print_error "Please set SWIFT_PATH environment variable or check installation"
        return 1
    fi
    print_success "SWIFT executable found: $SWIFT_PATH"
    
    # Check required Python packages
    local required_packages=("numpy" "h5py" "matplotlib" "woma")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            print_error "Python package '$package' not installed"
            return 1
        fi
    done
    print_success "All required Python packages are available"
    
    # Print directory information
    echo "Script directory: $SCRIPT_DIR"
    echo "Project root: $PROJECT_ROOT"
    echo "EoS Tables directory: $EOS_TABLES_DIR"
}

setup_eos_tables() {
    print_header "SETTING UP EQUATION OF STATE TABLES"
    
    # Check if EoSTables directory exists
    if ! check_directory_exists "$EOS_TABLES_DIR"; then
        print_error "EoSTables directory not found at: $EOS_TABLES_DIR"
        print_error "Expected location: $PROJECT_ROOT/EoSTables"
        print_error "Please download EoSTables to the correct location"
        return 1
    fi
    
    # Check for required EoS files
    local missing_files=()
    for eos_file in "${REQUIRED_EOS_FILES[@]}"; do
        if [[ ! -f "$EOS_TABLES_DIR/$eos_file" ]]; then
            missing_files+=("$eos_file")
        fi
    done
    
    if [[ ${#missing_files[@]} -eq 0 ]]; then
        print_success "All EoS tables are present"
        return 0
    fi
    
    print_warning "Missing EoS files: ${missing_files[*]}"
    
    # Try to download/get the tables
    if [[ -f "$EOS_TABLES_DIR/get_eos_tables.sh" ]]; then
        print_header "DOWNLOADING EOS TABLES"
        cd "$EOS_TABLES_DIR"
        if ! ./get_eos_tables.sh; then
            print_error "Failed to download EoS tables"
            cd - >/dev/null
            return 1
        fi
        cd - >/dev/null
    else
        print_error "get_eos_tables.sh not found in $EOS_TABLES_DIR"
        print_error "Please manually ensure the required EoS tables are present:"
        for file in "${missing_files[@]}"; do
            echo "  - $EOS_TABLES_DIR/$file"
        done
        return 1
    fi
    
    # Verify files were downloaded
    for eos_file in "${missing_files[@]}"; do
        if [[ ! -f "$EOS_TABLES_DIR/$eos_file" ]]; then
            print_error "EoS file still missing after download: $eos_file"
            return 1
        fi
    done
    
    print_success "All EoS tables successfully set up"
}

create_eos_symlinks() {
    print_header "CREATING EOS SYMLINKS FOR SWIFT"
    
    # Create symlinks in current directory so SWIFT can find the EoS files
    for eos_file in "${REQUIRED_EOS_FILES[@]}"; do
        local source_path="$EOS_TABLES_DIR/$eos_file"
        local link_name="./$eos_file"
        
        if [[ -f "$source_path" ]]; then
            if [[ -L "$link_name" ]]; then
                rm "$link_name"
            fi
            if ln -sf "$source_path" "$link_name"; then
                print_success "Created symlink: $link_name → $source_path"
            else
                print_error "Failed to create symlink: $link_name"
                return 1
            fi
        else
            print_error "EoS file not found for symlinking: $source_path"
            return 1
        fi
    done
}

run_python_script() {
    local script_name="$1"
    local description="${2:-$script_name}"
    
    print_header "RUNNING: $description"
    
    if ! check_file_exists "$script_name"; then
        print_error "Python script not found: $script_name"
        return 1
    fi
    
    if ! python3 "$script_name"; then
        print_error "Python script failed: $script_name"
        return 1
    fi
    
    print_success "Completed: $description"
}

run_swift_simulation() {
    local config_file="$1"
    local output_file="$2"
    local description="$3"
    
    print_header "RUNNING SWIFT: $description"
    
    if ! check_file_exists "$config_file"; then
        print_error "SWIFT config file not found: $config_file"
        return 1
    fi
    
    # Update the YML file to use correct EoS paths
    update_eos_paths_in_yml "$config_file"
    
    print_success "Starting simulation: $config_file"
    print_success "Output will be saved to: $output_file"
    
    # Run SWIFT with timeout (6 hours) and log output
    if timeout 6h "$SWIFT_PATH" --hydro --self-gravity --threads=28 "$config_file" 2>&1 | tee "$output_file"; then
        print_success "SWIFT simulation completed: $description"
    else
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            print_error "SWIFT simulation timed out after 6 hours: $description"
        else
            print_error "SWIFT simulation failed with exit code $exit_code: $description"
        fi
        return 1
    fi
}

update_eos_paths_in_yml() {
    local yml_file="$1"
    
    # Check if the YML file has relative paths that need updating
    if grep -q "planetary_ANEOS_forsterite_table_file:.*\.\." "$yml_file"; then
        print_warning "Updating EoS paths in $yml_file"
        
        # Create backup
        cp "$yml_file" "${yml_file}.backup"
        
        # Update paths to use absolute paths
        sed -i "s|planetary_ANEOS_forsterite_table_file:.*|planetary_ANEOS_forsterite_table_file: $EOS_TABLES_DIR/ANEOS_forsterite_S19.txt|g" "$yml_file"
        sed -i "s|planetary_ANEOS_Fe85Si15_table_file:.*|planetary_ANEOS_Fe85Si15_table_file: $EOS_TABLES_DIR/ANEOS_Fe85Si15_S20.txt|g" "$yml_file"
        
        print_success "Updated EoS paths to use absolute paths"
    fi
}

cleanup_on_error() {
    print_header "CLEANUP ON ERROR"
    print_warning "An error occurred. Cleaning up..."
    # Remove any symlinks we created
    for eos_file in "${REQUIRED_EOS_FILES[@]}"; do
        if [[ -L "./$eos_file" ]]; then
            rm -f "./$eos_file" && echo "Removed symlink: ./$eos_file"
        fi
    done
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================
main() {
    local error_occurred=0
    
    # Trap errors for cleanup
    trap 'error_occurred=1; cleanup_on_error' ERR
    
    print_header "STARTING SPIN IMPACT SIMULATION PIPELINE"
    echo "Started at: $(date)"
    echo "Working directory: $(pwd)"
    echo "Script location: $SCRIPT_DIR"
    
    # Step 1: Validate environment
    if ! validate_environment; then
        print_error "Environment validation failed"
        exit 1
    fi
    
    # Step 2: Setup EoS tables
    if ! setup_eos_tables; then
        exit 1
    fi
    
    # Step 3: Create EoS symlinks for SWIFT
    if ! create_eos_symlinks; then
        exit 1
    fi
    
    # Step 4: Create initial conditions
    if ! run_python_script "make_init_cond.py" "Creating initial conditions"; then
        exit 1
    fi
    
    # Step 5: Run target simulation
    if ! run_swift_simulation \
        "demo_target_${N_label}.yml" \
        "output_${N_label}_t.txt" \
        "Target settling simulation"; then
        exit 1
    fi
    
    # Step 6: Run impactor simulation  
    if ! run_swift_simulation \
        "demo_impactor_${N_label}.yml" \
        "output_${N_label}_i.txt" \
        "Impactor settling simulation"; then
        exit 1
    fi
    
    # Step 7: Generate plots
    if ! run_python_script "plot_snapshots.py" "Plotting snapshots"; then
        print_warning "Plotting failed, but continuing..."
        error_occurred=1
    fi
    
    if ! run_python_script "plot_profiles.py" "Plotting profiles"; then
        print_warning "Profile plotting failed, but continuing..."
        error_occurred=1
    fi
    
    # Step 8: Create impact initial conditions
    if ! run_python_script "make_impact_init_cond.py" "Creating impact initial conditions"; then
        exit 1
    fi
    
    # Cleanup symlinks
    cleanup_on_error
    
    # Final summary
    print_header "PIPELINE COMPLETED"
    echo "Finished at: $(date)"
    
    if [[ $error_occurred -eq 0 ]]; then
        print_success "All tasks completed successfully!"
        echo ""
        echo "Generated files:"
        echo "  - Initial conditions: demo_target_${N_label}.hdf5, demo_impactor_${N_label}.hdf5"
        echo "  - Simulation outputs: output_${N_label}_t.txt, output_${N_label}_i.txt"
        echo "  - Impact conditions: demo_impact_${N_label}.hdf5"
        echo "  - Plots: in current directory and snapshots/"
    else
        print_warning "Pipeline completed with warnings"
        echo "Some non-critical tasks failed, but main simulation completed."
    fi
    
    return $error_occurred
}

# Run main function
main "$@"