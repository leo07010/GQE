# GQE

This repository contains an executable example of the Generalized Quantum Eigensolver (GQE) configured for the nitrogen dimer (N₂), following the approach in [arXiv:2401.09253](https://arxiv.org/pdf/2401.09253).

## N₂ GQE experiment

The `gqe_n2.py` script builds the STO-3G Hamiltonian for N₂ at a 1.0977 Å bond length using the CUDA-Q chemistry utilities, fixes a 14-orbital (10-electron) active space, constructs a random operator pool of Pauli strings spanning the active qubits (size controlled by `--pool-size`), and runs the GQE adaptive loop. Each Pauli string is sampled site-by-site with at least one non-identity gate enforced and a random coefficient drawn from a configurable range, providing a reproducible yet diverse pool for the adaptive search. Deterministic seeds and deterministic Torch/cublas settings are enabled by default to improve reproducibility.

```bash
# CPU or single-GPU execution
python gqe_n2.py

# Multi-GPU execution via MPI
mpirun -np <num_ranks> python gqe_n2.py --mpi
```

Logged energies and optimization traces are written to `gqe_n2_logs/` via the Lightning Fabric CSV logger. Adjust `max_iters`, `ngates`, or geometry inside the script to explore different configurations.

### Handling non-finite gradients

If you encounter Fabric or PyTorch complaining about non-finite gradient norms during clipping on the first training step, the script patches `torch.nn.utils.clip_grad_norm_` to ignore the error, emit a warning, and zero any NaN/Inf gradients so CUDA kernels do not trip device-side assertions while training continues.
