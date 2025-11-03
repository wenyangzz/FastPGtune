# FastPGtune

<div align="center">    
</div>

---

FastPGtune is tested on a server configured with **Ubuntu 22.04.5** (Linux) and Python 3.11. It supports three graph index algorithms (NSG, HNSW, Vamana).



## Dependencies

1. Install Python 3.11 and necessary package [BoTorch](https://botorch.org/docs/getting_started).
2. Ensure build tools (e.g., `make`, `gcc`) are installed for compiling binary files.

## Complete Workflow

### Step 1: Generate Binary Files for Graph Index Algorithms

We support three graph index algorithms: NSG, HNSW, and Vamana. Generate their corresponding binary files using the provided `Makefile`:
- Navigate to the directory containing algorithm-specific `Makefile` (e.g., `algorithms/nsg/`, `algorithms/hnsw/`, `algorithms/vamana/`).
- Run the following command for each algorithm to compile the binary:

```bash
make  # Generates binary files (e.g., nsg_binary, hnsw_binary, vamana_binary)
```

Verify the binary files are generated in the target directory (default: bin/ under each algorithm's folder).

### Step 2: Update Binary File Paths in Utils

Modify the path to the generated binary files in `FastPGtune/auto-configure/FastPGtune/utils.py` to ensure the system correctly calls the binaries.

Example configuration for HNSW (update similar sections for NSG and Vamana):

```python
# In FastPGtune/auto-configure/FastPGtune/utils.py
PROJECT_DIR = "/path/to/your/FastPGtune/"  # Update to your project root path

# For HNSW binary (generated in Step 1)
print(f"{PROJECT_DIR}scripts/test_hnsw {cmd_args}")  # Verify command
os.system(f"{PROJECT_DIR}scripts/test_hnsw {cmd_args}")  # Execute binary with arguments
```

### Step 3: Configure Parameter Search Ranges

Specify the recommended parameter ranges for each algorithm using the corresponding JSON files:
- NSG: `auto-configure/whole_param_nsg.json`
- HNSW: `auto-configure/whole_param_hnsw.json`
- Vamana: `auto-configure/whole_param_vamana.json`

Example structure of a parameter range file (take `whole_param_hnsw.json` as reference):

```json
{
  "M": {"type": "int", "min": 8, "max": 64},
  "efConstruction": {"type": "int", "min": 100, "max": 2000},
  "ef": {"type": "int", "min": 10, "max": 500}
}
```

Adjust the ranges based on your performance requirements and hardware specifications.

### Step 4: Run Benchmark and Optimization

#### Benchmark (Baseline)

Run the baseline test for a specific algorithm. For example, for NSG:

```bash
# Navigate to the algorithm's test directory
cd FastPGtune/auto-configure/tests
python nsg_1.py  # Runs baseline with default parameters
```

#### Optimization with FastPGtune

Run the optimization script for the target algorithm to find optimal parameters. For example, for NSG:

```bash
# Navigate to the optimization directory
cd FastPGtune/auto-configure/optimization
python NSG_n.py  # Starts auto-tuning using parameters from whole_param_nsg.json
```

#### Logs and Results

Optimization results (parameter combinations, search speed, recall rate) are logged in real time to `record.log` and `pobo_record.log` under the running directory.

Example log entry:

```plaintext
[5] 42 {M: 32, efConstruction: 500, ef: 200} 185.23 0.9876  # {parameters} [speed (ms)] [recall rate]
```

## Troubleshooting

### Common Issues

1. **Binary not found errors**: Ensure the binary files are generated in Step 1 and the paths in `utils.py` are correctly set in Step 2.

2. **Build failures**: Verify that build tools (make, gcc) are installed and accessible in your PATH.

3. **Python package errors**: Ensure BoTorch and other required Python packages are installed for your Python 3.11 environment.

4. **Performance issues**: If optimization results are unsatisfactory, adjust parameter ranges in the JSON configuration files.

## Citation

If you use FastPGtune in your scientific article, please cite our ICDE 2024 paper:

```plaintext
@inproceedings{yang2024vdtuner,
     title={VDTuner: Automated Performance Tuning for Vector Data Management Systems},
     author={Yang, Tiannuo and Hu, Wen and Peng, Wangqi and Li, Yusen and Li, Jianguo and Wang, Gang and Liu, Xiaoguang},
     booktitle={2024 IEEE 40th International Conference on Data Engineering (ICDE)},
     year={2024}
}
```



