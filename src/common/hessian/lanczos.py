""" Use scipy/ARPACK implicitly restarted lanczos to find top k eigenthings """
from typing import Tuple

import numpy as np
import torch
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import svds as scipy_svds
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from warnings import warn

import src.common.hessian.utils as utils
from src.common.hessian.operator import Operator


def lanczos(
    operator: Operator,
    num_eigenthings: int =10,
    which: str ="LM",
    max_steps: int =20,
    tol: float =1e-1,
    num_lanczos_vectors: int =None,
    init_vec: np.ndarray =None,
    device: str ='cuda:0',
    fp16: bool =False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use the scipy.sparse.linalg.eigsh hook to the ARPACK lanczos algorithm
    to find the top k eigenvalues/eigenvectors.

    Please see scipy documentation for details on specific parameters
    such as 'which'.

    Parameters
    -------------
    operator: operator.Operator
        linear operator to solve.
    num_eigenthings : int
        number of eigenvalue/eigenvector pairs to compute
    which : str ['LM', SM', 'LA', SA']
        L,S = largest, smallest. M, A = in magnitude, algebriac
        SM = smallest in magnitude. LA = largest algebraic.
    max_steps : int
        maximum number of arnoldi updates
    tol : float
        relative accuracy of eigenvalues / stopping criterion
    num_lanczos_vectors : int
        number of lanczos vectors to compute. if None, > 2*num_eigenthings
        for stability.
    init_vec: [torch.Tensor, torch.cuda.Tensor]
        if None, use random tensor. this is the init vec for arnoldi updates.
    device: str
        device for calculating tensors
    fp16: bool
        if true, keep operator input/output in fp16 instead of fp32.

    Returns
    ----------------
    eigenvalues : np.ndarray
        array containing `num_eigenthings` eigenvalues of the operator
    eigenvectors : np.ndarray
        array containing `num_eigenthings` eigenvectors of the operator
    """
    if isinstance(operator.size, int):
        size = operator.size
    else:
        size = operator.size[0]
    shape = (size, size)

    if num_lanczos_vectors is None:
        num_lanczos_vectors = min(2 * num_eigenthings, size - 1)
    if num_lanczos_vectors < 2 * num_eigenthings:
        warn(
            "[lanczos] number of lanczos vectors should usually be > 2*num_eigenthings"
        )

    def _scipy_apply(x):
        x = torch.from_numpy(x)
        x = utils.maybe_fp16(x, fp16)
        if device != 'cpu':
            x = x.to(device)
        out = operator.apply(x)
        out = utils.maybe_fp16(out, fp16)
        out = out.cpu().numpy()
        return out

    scipy_op = ScipyLinearOperator(shape, _scipy_apply)
    if init_vec is None:
        init_vec = np.random.rand(size)

    eigenvals, eigenvecs = linalg.eigsh(
        A=scipy_op,
        k=num_eigenthings,
        which=which,
        maxiter=max_steps,
        tol=tol,
        ncv=num_lanczos_vectors,
        return_eigenvectors=True,
    )
    return eigenvals, eigenvecs.T


def lanczos_svd(
    operator: Operator,
    num_singularthings: int = 10,
    max_steps: int = 20,
    tol: float = 1e-1,
    num_lanczos_vectors: int = None,
    init_vec: np.ndarray = None,
    device: str = 'cuda:0',
    fp16: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use the scipy.sparse.linalg.svds hook to the ARPACK lanczos algorithm
    to find the top k singular values/vectors.

    Parameters
    -------------
    operator: operator.Operator
        linear operator to solve.
    num_singularthings : int
        number of singular values/vectors to compute.
    max_steps : int
        maximum number of lanczos updates.
    tol : float
        relative accuracy of singular values / stopping criterion.
    num_lanczos_vectors : int
        number of lanczos vectors to compute. if None, > 2*num_singularthings for stability.
    init_vec: [torch.Tensor, torch.cuda.Tensor]
        if None, use random tensor. this is the init vec for lanczos updates.
    device: str
        device for calculating tensors
    fp16: bool
        if true, keep operator input/output in fp16 instead of fp32.

    Returns
    ----------------
    singular_values : np.ndarray
        array containing `num_singularthings` singular values of the operator.
    left_singular_vectors : np.ndarray
        array containing `num_singularthings` left singular vectors of the operator.
    right_singular_vectors : np.ndarray
        array containing `num_singularthings` right singular vectors of the operator.
    """
    if isinstance(operator.size, int):
        size = operator.size
    else:
        size = operator.size[0]
    shape = (size, size)

    if num_lanczos_vectors is None:
        num_lanczos_vectors = min(2 * num_singularthings, size - 1)
    if num_lanczos_vectors < 2 * num_singularthings:
        warn("[lanczos_svd] number of lanczos vectors should usually be > 2*num_singularthings")

    def _scipy_apply(x):
        x = torch.from_numpy(x)
        x = utils.maybe_fp16(x, fp16)
        if device != 'cpu':
            x = x.to(device)
        out = operator.apply(x)
        out = utils.maybe_fp16(out, fp16)
        out = out.cpu().numpy()
        return out
    
    def _scipy_rapply(x):
        return _scipy_apply(x)

    scipy_op = ScipyLinearOperator(shape, matvec=_scipy_apply, rmatvec=_scipy_rapply)

    if init_vec is None:
        init_vec = np.random.rand(size)

    singular_values, left_singular_vectors, right_singular_vectors = scipy_svds(
        A=scipy_op,
        k=num_singularthings,
        which='LM',
        maxiter=max_steps,
        tol=tol,
        ncv=num_lanczos_vectors,
        v0=init_vec,
        return_singular_vectors=True
    )
    
    return singular_values, left_singular_vectors.T, right_singular_vectors.T