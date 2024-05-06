import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
        # hook functions for analysis
        self.hook_enabled = False
        self._activations = {}
        self._register_hooks()

    def encode(self, x):
        b = self.backbone(x)

        return b     

    def forward(self, x):
        """
        [backbone]: (n,c,h,w)-> Tuple((n,c,h,w), info)
        [head]: (n,c,h,w)-> Tuple((n,d), info)
        """
        b = self.backbone(x)
        h = self.head(b)

        return h
    
    def get_activations(self):
        """ 
        return activations after a forward pass 
        """
        return self._activations
    
    def _register_hooks(self):
        def hook_fn(module, input, output, name):
            if self.hook_enabled:
                self._activations[name] = output

        def register_hook(module, prefix=''):
            for name, sub_module in module.named_children():
                sub_prefix = f'{prefix}_{name}' if prefix else name
                if list(sub_module.children()):  # Check if it is a composite module
                    register_hook(sub_module, sub_prefix)
                else:
                    hook_name = f'{prefix}_{name}' if prefix else name
                    sub_module.register_forward_hook(
                        lambda module, input, output, name=hook_name: hook_fn(module, input, output, name)
                    )
        register_hook(self.backbone, 'backbone')
        register_hook(self.head, 'head')

    def enable_hooks(self):
        self.hook_enabled = True
        self._activations.clear()

    def disable_hooks(self):
        self.hook_enabled = False
        self._activations.clear()