'''
Modified Hessian computation in PyTorch.

Modifications
[1] Include computing top-k singular values

Reference:
[1] https://github.com/noahgolmant/pytorch-hessian-eigenthings
'''
from src.common.hessian.power_iter import power_iteration, deflated_power_iteration
from src.common.hessian.power_iter import svd_power_iteration, deflated_svd_power_iteration
from src.common.hessian.lanczos import lanczos, lanczos_svd
from src.common.hessian.hvp_operator import HVPOperator


def compute_hessian_eigen_things(
    model,
    dataloader,
    loss,
    num_vals=10,
    full_dataset=True,
    mode="power_iter",
    device='cuda:0',
    fp16=False,
    max_possible_gpu_samples=2 ** 16,
    **kwargs
):
    """
    Computes the top `num_eigenthings` eigenvalues and eigenvecs
    for the hessian of the given model by using subsampled power iteration
    with deflation and the hessian-vector product

    Parameters
    ---------------

    model : Module
        pytorch model for this netowrk
    dataloader : torch.data.DataLoader
        dataloader with x,y pairs for which we compute the loss.
    loss : torch.nn.modules.Loss | torch.nn.functional criterion
        loss function to differentiate through
    num_vals : int
        number of eigenvalues/eigenvecs to compute. computed in order of
        decreasing eigenvalue magnitude.
    full_dataset : boolean
        if true, each power iteration call evaluates the gradient over the
        whole dataset.
        (if False, you might want to check if the eigenvalue estimate variance
         depends on batch size)
    mode : str ['power_iter', 'lanczos']
        which backend algorithm to use to compute the top eigenvalues.
    device:
        attempt to use the device for all lin alg computatoins
    fp16: bool
        if true, store and do math with eigenvectors, gradients, etc. in fp16.
        (you should test if this is numerically stable for your application)
    max_possible_gpu_samples:
        the maximum number of samples that can fit on-memory. used
        to accumulate gradients for large batches.
        (note: if smaller than dataloader batch size, this can have odd
         interactions with batch norm statistics)
    **kwargs:
        contains additional parameters passed onto lanczos or power_iter.
    """
    hvp_operator = HVPOperator(
        model,
        dataloader,
        loss,
        device=device,
        full_dataset=full_dataset,
        max_possible_gpu_samples=max_possible_gpu_samples,
    )
    eigen_vals, eigen_vecs = None, None
    if mode == "power_iter":
        eigen_vals, eigen_vecs = deflated_power_iteration(
            hvp_operator, num_vals, device=device, fp16=fp16, **kwargs
        )
    elif mode == "lanczos":
        eigen_vals, eigen_vecs = lanczos(
            hvp_operator, num_vals, device=device, fp16=fp16, **kwargs
        )
    else:
        raise ValueError("Unsupported mode %s (must be power_iter or lanczos)" % mode)
    return eigen_vals, eigen_vecs


def compute_hessian_singular_things(
    model,
    dataloader,
    loss,
    num_vals=10,
    full_dataset=True,
    mode="power_iter",
    device='cuda:0',
    fp16=False,
    max_possible_gpu_samples=2 ** 16,
    **kwargs
):
    """
    Computes the top `num_singularthings` singularvalues a
    for the hessian of the given model by using subsampled power iteration
    with deflation and the hessian-vector product

    Parameters
    ---------------

    model : Module
        pytorch model for this netowrk
    dataloader : torch.data.DataLoader
        dataloader with x,y pairs for which we compute the loss.
    loss : torch.nn.modules.Loss | torch.nn.functional criterion
        loss function to differentiate through
    num_vals : int
        number of singularvalues to compute. computed in order of
        decreasing singular value magnitude.
    full_dataset : boolean
        if true, each power iteration call evaluates the gradient over the
        whole dataset.
        (if False, you might want to check if the singular value estimate variance
         depends on batch size)
    mode : str ['power_iter', 'lanczos']
        which backend algorithm to use to compute the top singular values.
    device:
        attempt to use the device for all lin alg computatoins
    fp16: bool
        if true, store and do math with singular vectors, gradients, etc. in fp16.
        (you should test if this is numerically stable for your application)
    max_possible_gpu_samples:
        the maximum number of samples that can fit on-memory. used
        to accumulate gradients for large batches.
        (note: if smaller than dataloader batch size, this can have odd
         interactions with batch norm statistics)
    **kwargs:
        contains additional parameters passed onto lanczos or power_iter.
    """
    hvp_operator = HVPOperator(
        model,
        dataloader,
        loss,
        device=device,
        full_dataset=full_dataset,
        max_possible_gpu_samples=max_possible_gpu_samples,
    )
    singular_vals, left_singular_vecs, right_singular_vecs = None, None, None
    if mode == "power_iter":
        singular_vals, left_singular_vecs, right_singular_vecs = deflated_svd_power_iteration(
            hvp_operator, num_vals, device=device, fp16=fp16, **kwargs
        )
    elif mode == "lanczos":
        singular_vals, left_singular_vecs, right_singular_vecs = lanczos_svd(
            hvp_operator, num_vals, device=device, fp16=fp16, **kwargs
        )
    else:
        raise ValueError("Unsupported mode %s (must be power_iter or lanczos)" % mode)
    return singular_vals, left_singular_vecs, right_singular_vecs


__all__ = [
    "power_iteration",
    "svd_power_iteration",
    "deflated_power_iteration",
    "deflated_svd_power_iteration",
    "lanczos",
    "lanczos_svd",
    "HVPOperator",
    "compute_hessian_eigen_things",
    "compute_hessian_singular_things",
]
