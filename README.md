# Quantization of Low-Rank Gradient Projection Methods for LLM Training

This project forks and expands on the pre-release version of the GaLore algorithm, proposed by [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507) (published in ICML 2024). Gradient Low-Rank Projection (GaLore) is a memory-efficient low-rank training strategy that allows *full-parameter* learning but is more *memory-efficient* than common low-rank adaptation methods, such as LoRA. In short, GaLore takes a low-rank approximation of the gradient matrix instead of the weight matrix, and uses this low-rank matrix when computing and storing optimizer states. In order to compute this low rank-approximation, the gradient matrix is multiplied by a projection matrix. In this project, we experiment with quantizing the projection matrix to save even more memory, and explore its effects on LLM fine-tuning performance — specifically tuning RoBERTa on GLUE tasks.

We aim to make an open-source contribution to the release of GaLore. In fact, the authors of GaLore have given us advice throughout this project.

## Implementing quantization for gradient projection matrices

`galore_torch/galore_projector.py` contains the main changes we made to the original release. We implement the option to quantize gradient projection matrices into fp8 or fp4, blockwise or full. 

## Benchmark: Fine-Tuning RoBERTa on GLUE tasks
`run_glue.py` is the main script for fine-tuning RoBERTa models on GLUE tasks with GaLore. The original script was developed by the authors of the GaLore paper, but we make minor changes for our own convenience (e.g. wandb logging).

`scripts/proj_quantize_glue` contains the scripts we wrote and ran for our experiments.

## Results
We ran the above benchmarks on NVIDIA L4 GPU with 24GB RAM and G2-standard-4 CPU (4 vCPU, 16GB RAM) on CUDA 11.8. We chose two GLUE tasks, MPRC and COLA and document their scores.

![alt text](https://github.com/seyoungree/hpml_galore/blob/quantize/imgs/results.png "chart")

The best hyperparameters are bolded, and we use these for comparison. We observed that using 8-bit blockwise quantization of the projection matrices yields slightly higher scores for both MPRC and COLA tasks compared to no quantization. Although this implies that 8-bit blockwise quantization has minimal effect on model performance here, we note that RoBERTa base is a relatively small language model compared to those that would actually leverage projection matrix quantization in practice. It's possible that we observe a more noticeable effect of quantization on larger models and/or larger (i.e. higher rank) projection matrices.

## Citation
```bibtex
@misc{zhao2024galore,
      title={GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection}, 
      author={Jiawei Zhao and Zhenyu Zhang and Beidi Chen and Zhangyang Wang and Anima Anandkumar and Yuandong Tian},
      year={2024},
      eprint={2403.03507},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
