# Quantum Convolutional Graph Neural Network

This repository contains the implementation of a Quantum Convolutional Graph Neural Network for node classification on citation networks. The code reproduces and extends the results reported in the paper by implementing two different approaches for node selection and classification.

## Requirements

Before running the code, please install all dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing Paper Results

To reproduce the results mentioned in the paper, follow these steps:

1. Navigate to the source directory:
   ```bash
   cd src/
   ```

2. Run the training script:
   ```bash
   python train_main_paper.py
   ```

This will generate test loss values and evaluation metrics (accuracy) that match the results reported in the paper for 1024 nodes. The average accuracy is approximately 72% reported in the original paper.

## Approaches Implemented

This repository implements two different methodologies for the binary semi-supervised classification task on the Cora citation network dataset:

### 1. Original Paper Approach (Default)
- Randomly selects a required number of nodes from the Cora dataset
- Uses a quantum circuit feature extractor with depth 7
- Achieves approximately 71-72% accuracy as reported in the paper
- Implementation uses `data_prep_node_selection.py` for data preparation


### 2. My Approach
- Considers the entire dataset with class separation
- Requires only a quantum circuit of depth 1
- Achieves accuracy comparable to or better than the original approach
- Implementation uses `data_prep.py` for data preparation



## Switching Between Approaches

To check my approach approaches, please run the following:
   ```bash
   python train.py
   ```

## Important Note

Please do not modify the model dimensions or parameters in the source code. The quantum network dimensions must match the batch size and feature numbers. Changing these parameters may cause dimension mismatches and result in errors. The code is configured to reproduce the results as reported in the notebooks and essay.

## Results

Both approaches achieve an accuracy of approximately 71% on average, which aligns with the original paper's reported 72% accuracy. The alternative approach achieves this with a simpler quantum circuit (depth 1 vs depth 7).

## Files
- Jupyter notebooks are included with detailed analyses of the results for both approaches
