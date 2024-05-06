"""
This module contains functions to perform power iteration with deflation
to compute the top eigenvalues and eigenvectors of a linear operator
"""
from typing import Tuple

import numpy as np
import torch

from src.common.hessian.operator import Operator, LambdaOperator
import src.common.hessian.utils as utils


def deflated_power_iteration(
    operator: Operator,
    num_eigenthings: int = 10,
    power_iter_steps: int = 20,
    power_iter_err_threshold: float = 1e-4,
    momentum: float = 0.0,
    device='cuda:0',
    fp16: bool = False,
    to_numpy: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top k eigenvalues by repeatedly subtracting out dyads
    operator: linear operator that gives us access to matrix vector product
    num_eigenvals number of eigenvalues to compute
    power_iter_steps: number of steps per run of power iteration
    power_iter_err_threshold: early stopping threshold for power iteration
    returns: np.ndarray of top eigenvalues, np.ndarray of top eigenvectors
    """
    eigenvals = []
    eigenvecs = []
    current_op = operator
    prev_vec = None

    def _deflate(x, val, vec):
        return val * vec.dot(x) * vec

    utils.log("beginning deflated power iteration")
    for i in range(num_eigenthings):
        utils.log("computing eigenvalue/vector %d of %d" % (i + 1, num_eigenthings))
        eigenval, eigenvec = power_iteration(
            current_op,
            power_iter_steps,
            power_iter_err_threshold,
            momentum=momentum,
            device=device,
            fp16=fp16,
            init_vec=prev_vec,
        )
        utils.log("eigenvalue %d: %.4f" % (i + 1, eigenval))

        def _new_op_fn(x, op=current_op, val=eigenval, vec=eigenvec):
            return utils.maybe_fp16(op.apply(x), fp16) - _deflate(x, val, vec)

        current_op = LambdaOperator(_new_op_fn, operator.size)
        prev_vec = eigenvec
        eigenvals.append(eigenval)
        eigenvec = eigenvec.cpu()
        if to_numpy:
            # Clone so that power_iteration can continue to use torch.
            numpy_eigenvec = eigenvec.detach().clone().numpy()
            eigenvecs.append(numpy_eigenvec)
        else:
            eigenvecs.append(eigenvec)

    eigenvals = np.array(eigenvals)
    eigenvecs = np.array(eigenvecs)

    # sort them in descending order
    sorted_inds = np.argsort(eigenvals)
    eigenvals = eigenvals[sorted_inds][::-1]
    eigenvecs = eigenvecs[sorted_inds][::-1]
    return eigenvals, eigenvecs


def power_iteration(
    operator: Operator,
    steps: int = 20,
    error_threshold: float = 1e-4,
    momentum: float = 0.0,
    device: str = 'cuda:0',
    fp16: bool = False,
    init_vec: torch.Tensor = None,
    ) -> Tuple[float, torch.Tensor]:
    """
    Compute dominant eigenvalue/eigenvector of a matrix
    operator: linear Operator giving us matrix-vector product access
    steps: number of update steps to take
    returns: (principal eigenvalue, principal eigenvector) pair
    """
    vector_size = operator.size  # input dimension of operator
    if init_vec is None:
        vec = torch.rand(vector_size)
    else:
        vec = init_vec

    vec = utils.maybe_fp16(vec, fp16)

    if device != 'cpu':
        vec = vec.to(device)

    prev_lambda = 0.0
    prev_vec = utils.maybe_fp16(torch.randn_like(vec), fp16)
    for i in range(steps):
        prev_vec = vec / (torch.norm(vec) + 1e-6)
        new_vec = utils.maybe_fp16(operator.apply(vec), fp16) - momentum * prev_vec
        # need to handle case where we end up in the nullspace of the operator.
        # in this case, we are done.
        if torch.norm(new_vec).item() == 0.0:
            return 0.0, new_vec
        lambda_estimate = vec.dot(new_vec).item()
        diff = lambda_estimate - prev_lambda
        vec = new_vec.detach() / torch.norm(new_vec)
        if lambda_estimate == 0.0:  # for low-rank
            error = 1.0
        else:
            error = np.abs(diff / lambda_estimate)
        utils.progress_bar(i, steps, "power iter error: %.4f" % error)
        if error < error_threshold:
            break
        prev_lambda = lambda_estimate
    return lambda_estimate, vec


def deflated_svd_power_iteration(
    operator: Operator,
    num_singularthings: int = 10,
    svd_power_iter_steps: int = 20,
    svd_power_iter_err_threshold: float = 1e-4,
    device: str = 'cuda:0',
    fp16: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    singular_values = []
    left_singular_vectors = []
    right_singular_vectors = []
    current_op = operator

    for i in range(num_singularthings):
        singular_value, left_singular_vector, right_singular_vector = svd_power_iteration(
            current_op,
            svd_power_iter_steps,
            svd_power_iter_err_threshold,
            device=device,
            fp16=fp16
        )

        def _new_op_fn(x, op=current_op, val=singular_value, left_vec=left_singular_vector, right_vec=right_singular_vector):
            return op.apply(x) - val * (right_vec @ x) * left_vec
        
        current_op = LambdaOperator(_new_op_fn, operator.size)
        singular_values.append(singular_value)
        left_singular_vectors.append(left_singular_vector.cpu().numpy())
        right_singular_vectors.append(right_singular_vector.cpu().numpy())
        
    singular_values = np.array(singular_values)
    left_singular_vectors = np.array(left_singular_vectors)
    right_singular_vectors = np.array(right_singular_vectors)
        
    # sort them in descending order
    sorted_inds = np.argsort(singular_values)
    singular_values = singular_values[sorted_inds][::-1]
    left_singular_vectors = left_singular_vectors[sorted_inds][::-1]
    right_singular_vectors = right_singular_vectors[sorted_inds][::-1]

    return singular_values, left_singular_vectors, right_singular_vectors


def svd_power_iteration(
    operator: Operator,
    steps: int = 20,
    error_threshold: float = 1e-4,
    device: str = 'cuda:0',
    fp16: bool = False
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        
    vector_size = operator.size
    u = torch.rand(vector_size)
    v = torch.rand(vector_size)
    
    # Normalize the initial vectors
    u = u / torch.norm(u)
    v = v / torch.norm(v)

    u = utils.maybe_fp16(u, fp16)
    v = utils.maybe_fp16(v, fp16)

    if device != 'cpu':
        u = u.to(device)
        v = v.to(device)
        
    prev_u = torch.zeros_like(u)
    prev_v = torch.zeros_like(v)

    for i in range(steps):
        # Hessian-vector product (H * v)
        Hv = operator.apply(v) # + 1e-5 * v
        u_new = Hv / (torch.norm(Hv) + 1e-6)

        # Transpose Hessian-vector product (H.T * u)
        Ht_u = operator.apply(u) # + 1e-5 * u
        v_new = Ht_u / (torch.norm(Ht_u) + 1e-6)

        singval_estimate = torch.norm(Hv).item()
        error = np.abs(singval_estimate - torch.norm(u).item()) / (singval_estimate + 1e-6)
        utils.progress_bar(i, steps, f"SVD Power iter error: %.4f singval: %.4f" % (error, singval_estimate))

        if error < error_threshold:
            break

        prev_u, prev_v = u_new, v_new
        u, v = u_new, v_new
        
    singular_value = torch.norm(Hv).item()
    return singular_value, u, v