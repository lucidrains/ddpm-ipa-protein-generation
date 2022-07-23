## DDPM + IPA for Protein Structure and Sequence Generation (wip)

Implementation of the DDPM + IPA (invariant point attention) for protein generation, as outlined in the paper <a href="https://arxiv.org/abs/2205.15019">Protein Structure and Sequence Generation with Equivariant Denoising Diffusion Probabilistic Models</a>. They basically combined the <a href="https://github.com/lucidrains/invariant-point-attention">invariant point attention</a> module from <a href="https://github.com/deepmind/alphafold">Alphafold2</a> (used for coordinate refinement) with a standard DDPM, and demonstrate very cool infilling capabilities for protein structure generation.

I will also equip this with ability to condition on encoded text, identical to <a href="https://github.com/lucidrains/imagen-pytorch">Imagen</a>. Eventually, I will also try to offer a version using <a href="https://arxiv.org/abs/2107.07675">Insertion-deletion DDPM</a> (but I have yet to replicate this work and open source it)

## Citations

```bibtex
@misc{https://doi.org/10.48550/arxiv.2205.15019,
  doi     = {10.48550/ARXIV.2205.15019},
  url     = {https://arxiv.org/abs/2205.15019},
  author  = {Anand, Namrata and Achim, Tudor},
  keywords = {Quantitative Methods (q-bio.QM), Artificial Intelligence (cs.AI), FOS: Biological sciences, FOS: Biological sciences, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title   = {Protein Structure and Sequence Generation with Equivariant Denoising Diffusion Probabilistic Models},
  publisher = {arXiv},
  year      = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
