# FastPGtune
<div align="center">    

</div>

---

FastPGtune is tested on a server configured with **Ubuntu 22.04.5** (Linux) and Python 3.11. 

## Dependencies
1. Install Python 3.11 and necessary package [BoTorch](https://botorch.org/docs/getting_started).
   
## Preparations
#### Modify the default engine in benchmark.  
- Adjust the default engine settings in `vector-db-benchmark` according to your target vector database configuration.  
- After configuration, test the benchmark with a small dataset to verify deployment. Go to `vector-db-benchmark-master` and run:
  ```bash
  sudo ./run_engine.sh "" "" random-100
