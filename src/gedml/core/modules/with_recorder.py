import torch

from ...config.setting.recorder_setting import TO_RECORD_LIST

class WithRecorder(torch.nn.Module):
    """
    Work with ``recorder protocol``.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        setattr(self, TO_RECORD_LIST, [])
    
    def preprocess_hook(self, trainer):
        pass

    def callback_hook(self, trainer):
        pass
    
    def add_recordable_attr(self, name: str):
        """
        Args:
            name (str):
                The name of attributes.
        """
        # add list
        if not getattr(self, TO_RECORD_LIST, None):
            setattr(self, TO_RECORD_LIST, [])
        # add attr
        to_record_list = getattr(self, TO_RECORD_LIST)
        if name not in to_record_list:
            to_record_list.append(name)
        if not getattr(self, name, None):
            setattr(self, name, 0)
