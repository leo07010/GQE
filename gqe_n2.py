import argparse
import os
import random

import cudaq
import cudaq_solvers as solvers
import torch
from torch.nn import utils as torch_utils
from cudaq import spin
from lightning.fabric.loggers import CSVLogger

from cudaq_solvers.gqe_algorithm.gqe import get_default_config


parser = argparse.ArgumentParser()
parser.add_argument("--mpi", action="store_true")
parser.add_argument(
    "--pool-size",
    type=int,
    default=200,
    help="Total number of random Pauli strings to include in the operator pool.",
)
args = parser.parse_args()

if args.mpi:
    try:
        cudaq.set_target("nvidia", option="mqpu")
        cudaq.mpi.initialize()
    except RuntimeError:
        print(
            "Warning: NVIDIA GPUs or MPI not available, unable to use CUDA-Q MQPU. Skipping..."
        )
        raise SystemExit(0)
else:
    try:
        cudaq.set_target("nvidia", option="fp64")
    except RuntimeError:
        cudaq.set_target("qpp-cpu")

# Set deterministic seed and environment variables for deterministic behavior
# Disable this section for non-deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(3047)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(3047)


_ORIG_CLIP_GRAD_NORM = torch_utils.clip_grad_norm_


def _safe_clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float | int = 2.0,
    error_if_nonfinite: bool = False,
    **kwargs,
):
    """Clip gradients but tolerate non-finite norms.

    Lightning's Fabric utilities call ``torch.nn.utils.clip_grad_norm_`` during
    training. In some environments the random initialization can yield
    non-finite gradient norms on the first step, which raises and aborts
    training. This shim disables the error-on-nonfinite behavior while still
    returning the computed norm so Fabric can continue. A warning is emitted
    if the norm is not finite so the training logs surface the issue. The
    function mirrors the upstream signature (including ``error_if_nonfinite``
    and extra keyword arguments) to remain compatible with Lightning.
    """

    norm = _ORIG_CLIP_GRAD_NORM(
        parameters,
        max_norm,
        norm_type=norm_type,
        error_if_nonfinite=False,
        **kwargs,
    )
    if not torch.isfinite(norm):
        print(
            "Warning: non-finite gradient norm detected during clipping; zeroing NaN/Inf gradients to avoid CUDA assertion failures."
        )

        # Sanitise any non-finite gradient values in-place to prevent downstream
        # CUDA kernels (e.g., probability computations) from tripping device-side
        # asserts.
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        for param in parameters:
            if param is None:
                continue
            grad = param.grad
            if grad is None:
                continue

            # Replace NaN/Inf entries with zero to keep kernels well-defined.
            if torch.isfinite(grad).all():
                continue

            finite_mask = torch.isfinite(grad)
            if not finite_mask.all():
                grad.data = torch.where(finite_mask, grad, torch.zeros_like(grad))

    return norm


# Apply the shim globally so Fabric's gradient clipping remains tolerant.
torch_utils.clip_grad_norm_ = _safe_clip_grad_norm_

# Create the molecular Hamiltonian for N2 using a 14-orbital active space
geometry = [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.0977))]
ACTIVE_SPATIAL_ORBITALS = 14
ACTIVE_ELECTRONS = 10
molecule = solvers.create_molecule(
    geometry,
    "sto-3g",
    0,
    0,
    casci=True,
    active_space=(ACTIVE_SPATIAL_ORBITALS, ACTIVE_ELECTRONS),
)

spin_ham = molecule.hamiltonian
n_qubits = ACTIVE_SPATIAL_ORBITALS * 2
n_electrons = ACTIVE_ELECTRONS

def build_operator_pool(
    n_spin_orbitals: int,
    num_pauli_strings: int,
    coeff_range: tuple[float, float] = (-0.1, 0.1),
    seed: int | None = 3047,
) -> list[cudaq.SpinOperator]:
    """Generate a randomized operator pool sized to the active qubits.

    Each Pauli string is sampled site-by-site across ``n_spin_orbitals`` using
    the provided seed for reproducibility. At least one non-identity gate is
    forced per string (by converting the final site to ``X`` when needed), and
    every string receives a uniformly sampled coefficient from ``coeff_range``.
    ``num_pauli_strings`` controls how many of these coefficient-weighted
    strings are returned to the GQE driver.
    """
    if n_spin_orbitals < 1:
        raise ValueError("Operator pool requires at least one qubit.")
    if num_pauli_strings < 1:
        raise ValueError("num_pauli_strings must be positive.")

    rng = random.Random(seed)
    pauli_gates = [spin.x, spin.y, spin.z, spin.i]

    def random_pauli_operator() -> cudaq.SpinOperator:
        op_expr = None
        has_non_identity = False
        for idx in range(n_spin_orbitals):
            gate = rng.choice(pauli_gates)
            if idx == n_spin_orbitals - 1 and not has_non_identity:
                gate = spin.x
            if gate is not spin.i:
                has_non_identity = True
            term = gate(idx)
            op_expr = term if op_expr is None else op_expr * term

        return cudaq.SpinOperator(op_expr)

    operator_pool: list[cudaq.SpinOperator] = []
    for _ in range(num_pauli_strings):
        coefficient = rng.uniform(*coeff_range)
        operator_pool.append(coefficient * random_pauli_operator())

    return operator_pool


def term_coefficients(op: cudaq.SpinOperator) -> list[complex]:
    return [term.evaluate_coefficient() for term in op]


def term_words(op: cudaq.SpinOperator) -> list[cudaq.pauli_word]:
    return [term.get_pauli_word(n_qubits) for term in op]


@cudaq.kernel
def kernel(
    n_qubits: int,
    n_electrons: int,
    coeffs: list[float],
    words: list[cudaq.pauli_word],
):
    q = cudaq.qvector(n_qubits)

    for i in range(n_electrons):
        x(q[i])

    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])


def cost(sampled_ops: list[cudaq.SpinOperator], **kwargs):
    full_coeffs = []
    full_words = []

    for op in sampled_ops:
        full_coeffs += [c.real for c in term_coefficients(op)]
        full_words += term_words(op)

    if args.mpi:
        handle = cudaq.observe_async(
            kernel,
            spin_ham,
            n_qubits,
            n_electrons,
            full_coeffs,
            full_words,
            qpu_id=kwargs["qpu_id"],
        )
        return handle, lambda res: res.get().expectation()
    else:
        return cudaq.observe(
            kernel, spin_ham, n_qubits, n_electrons, full_coeffs, full_words
        ).expectation()


op_pool = build_operator_pool(n_qubits, args.pool_size)

# Configure GQE
cfg = get_default_config()
cfg.use_fabric_logging = False
logger = CSVLogger("gqe_n2_logs", name="gqe")
cfg.fabric_logger = logger
cfg.save_trajectory = False
cfg.verbose = True

# Run GQE
minE, best_ops = solvers.gqe(cost, op_pool, max_iters=50, ngates=20, config=cfg)

# Only print results from rank 0 when using MPI
if (not args.mpi) or cudaq.mpi.rank() == 0:
    print(f"Ground Energy = {minE}")
    print("Ansatz Ops")
    for idx in best_ops:
        term = next(iter(op_pool[idx]))
        print(term.evaluate_coefficient().real, term.get_pauli_word(n_qubits))

if args.mpi:
    cudaq.mpi.finalize()
