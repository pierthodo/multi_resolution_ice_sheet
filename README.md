# Calculating Exposure to Extreme Sea Level Risk Requires High-Resolution Ice Sheet Models

[![DOI](https://zenodo.org/badge/697273383.svg)](https://doi.org/10.5281/zenodo.14392508)

This repository contains the code to reproduce the figures in the paper **"Calculating Exposure to Extreme Sea Level Risk Requires High-Resolution Ice Sheet Models"**.

## Repository Structure

- **`assets/ice_sheet_simulation`**: Contains the ice-sheet model runs used in the study.
- **`notebook/`**: Contains the Jupyter notebook to reproduce the figures.

## Installation

To set up the environment, install the repository with:

```bash
pip install -e .
```

## Running the Code

To reproduce the figures from the paper, navigate to the `notebook` directory and execute the Jupyter notebook. Ensure that all dependencies are installed and the required assets are available.

### Instructions

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook notebook/your_notebook_name.ipynb
   ```
2. Select the notebook file and follow the step-by-step instructions provided within the notebook cells to execute the simulations and generate the figures.

3. Modify parameters at the top of the notebook as needed to customize the simulation settings:
   - **`sample_size`**: Adjust the number of data samples for the simulation. Larger values result in higher accuracy but longer runtimes.
   - **`num_rep`**: Set the number of repetitions for Monte Carlo simulations or similar stochastic methods.

4. Run the notebook cells sequentially to ensure dependencies between cells are resolved correctly.

5. Once the notebook has completed execution, locate the generated plots in the `assets/plots` directory.

### Performance Considerations

Some notebooks involve computationally intensive simulations that may take several hours to run. While the notebooks currently run sequentially, performance could be significantly improved by adding parallel processing:

- The code can be modified to use the `pathos` library for multiprocessing support
- This would allow computationally heavy sections to run concurrently across multiple CPU cores
- Consider implementing this optimization if runtime is a concern for your use case

