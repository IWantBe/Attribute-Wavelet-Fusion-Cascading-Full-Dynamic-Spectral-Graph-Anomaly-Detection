# FDSGAD: Full-Dynamic Spectral Graph Anomaly Detection

Official implementation of

> **Attribute-Wavelet-Fusion Cascading: Full-Dynamic Spectral Graph Anomaly Detection**  
> Jianbo Zheng, Fangrong Zheng, Xiaoge Li, Jin Zhang, Chao Yang, Xiao-Ming Wu

FDSGAD extends our earlier work **DSGAD** (*Dynamic Spectral Graph Anomaly Detection*, AAAI 2025) with a dynamic attribute transformation module. The framework replaces the static three-stage pipeline of spectral GNN-based anomaly detectors (attribute transformation, graph wavelet extraction, feature fusion) with three data-adaptive components:

1. **Dynamic Attribute Transformation (DAT)** maps node attributes into a Fourier feature space with trainable Fourier frequencies and applies gated residual fusion.
2. **Dynamic Wavelets (DW)** combine the handcrafted Beta mother wavelet with trainable Beta-mixture wavelets to extract wavelet-specific features.
3. **Dynamic Fusion (DF)** aligns wavelet-specific features with wavelet-specific feature extraction (WFE) and fuses them with channel convolution fusion (CCF).

The fused representation is fed into a binary classifier trained with the weighted cross-entropy loss.

## Environment

The code was run with Python 3.10, PyTorch 2.0.0 (CUDA 11.8) and DGL 2.0.0 on a single GPU with 48 GB memory.

```bash
conda create -n fdsgad python=3.10 -y
conda activate fdsgad
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install dgl==2.0.0 -f https://data.dgl.ai/wheels/cu118/repo.html
pip install numpy scipy sympy scikit-learn pandas torch-geometric rtdl_num_embeddings
```

## Citation

```bibtex
@article{zheng2026fdsgad,
  title   = {Attribute-Wavelet-Fusion Cascading: Full-Dynamic Spectral Graph Anomaly Detection},
  author  = {Zheng, Jianbo and Zheng, Fangrong and Li, Xiaoge and Zhang, Jin and Yang, Chao and Wu, Xiao-Ming},
  year    = {2026}
}

@inproceedings{zheng2025dynamic,
  title     = {Dynamic Spectral Graph Anomaly Detection},
  author    = {Zheng, Jianbo and others},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025}
}
```
