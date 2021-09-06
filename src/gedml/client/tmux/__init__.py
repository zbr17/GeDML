"""
Note: 
    **Tmux** software support is required.

This module is designed for assisting distributed trainning and hyperparameter searching. There are two situations where this module can be used:

1. **Distributed training**. PyTorch distributed API need multi-device support. Tmux software is used for launching multi processes.

2. **Hyperparameter searching**. Tmux software is used for launching each experiment which corresponds to a certain hyperparameter combination.

More information about `Tmux <https://github.com/tmux/tmux>`_.
"""

from .manager import (
    TmuxManager,
    clear_tmux
)