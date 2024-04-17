# ViT Training Benchmark with ColossalAI
This is the NUS CS5260 course project, https://github.com/KoalaYuFeng/vit_train_benchmark_with_Colossalai. In the repository's `README.md` file, provide the following information:

## Model Used in the Experiment
In this repository, we utilize pretrained weights of the Vision Transformer (ViT) loaded from HuggingFace. We adapt the ViT training code to work with ColossalAI by leveraging the Boosting API, which is loaded with a chosen plugin. Each plugin corresponds to a specific type of training strategy. This example supports plugins including:
- `TorchDDPPlugin` (DDP)
- `LowLevelZeroPlugin` (Zero1/Zero2)
- `GeminiPlugin` (Gemini)

## Dataset Employed
We use the `BeansDataset` from HuggingFace.

## Instructions on How to Run the Code
1. First, ensure the correct version of PyTorch is installed that matches your CUDA version. In my case, with CUDA version 11.7, I install `torch 1.13.0`.
2. Include the requirements in the `requirements.txt`. You can install them using the command:
   ```bash
   pip install -r requirements.txt
3. Clone the ColossalAI repository from GitHub:
   ```bash
   git clone --recursive https://github.com/hpcaitech/ColossalAI.git
4. Navigate to the directory
   ```bash
   cd ColossalAI/examples/images/vit
5. Run the script:
   ```bash
   bash run_demo.sh // for training ViT:
   bash run_benchmark.sh // for benchmark ViT:

## Experiment Results

### Training Accuracy

| Epoch | Average Loss | Accuracy  |
|-------|--------------|-----------|
| 1     | 1.1607       | 85.94%    |
| 2     | 0.2364       | 97.66%    |
| 3     | 0.2099       | 98.44%    |

## Benchmark Results

The benchmarking was conducted using different plugins and batch sizes. The results are summarized in the table below:

| Plugin           | Batch Size per GPU | Throughput (samples/sec) | Maximum Memory Usage per GPU |
|------------------|--------------------|--------------------------|------------------------------|
| `torch_ddp`      | 8                  | 43.7168                  | 1.80 GB                      |
| `torch_ddp_fp16` | 8                  | 60.1283                  | 1.91 GB                      |
| `low_level_zero` | 8                  | 47.1534                  | 1.65 GB                      |
| `gemini`         | 8                  | 28.0425                  | 663.17 MB                    |
| `torch_ddp`      | 32                 | 66.7630                  | 2.34 GB                      |
| `torch_ddp_fp16` | 32                 | 153.6898                 | 2.25 GB                      |
| `low_level_zero` | 32                 | 143.5798                 | 1.66 GB                      |
| `gemini`         | 32                 | 110.6582                 | 663.17 MB                    |

For more detailed configurations and complete benchmark results, please refer to the log file in the repository.

