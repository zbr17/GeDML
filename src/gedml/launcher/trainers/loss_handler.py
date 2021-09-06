import torch
from collections import defaultdict

class LossHandler:
    def __init__(self):
        self.weight_values = {}
        self.loss_values = {"total": torch.Tensor(0)}
        self.avg_loss_list = defaultdict(list)
        self.avg_loss_values = {}

    def get_total(self, is_avg_value=False):
        if is_avg_value:
            return self.avg_loss_values["total"]
        else:
            return self.loss_values["total"] 
    
    def backward(self):
        self.loss_values["total"].backward()
    
    def zero_losses(self):
        for k in self.loss_values.keys():
            self.loss_values[k] = 0
    
    def record_losses(self, recorder, total_iterations):
        recorder.update(
            {"loss_values": self.loss_values},
            total_iterations
        )
        recorder.update(
            {"loss_weight": self.weight_values}, 
            total_iterations
        )
        
    def update_losses(self, loss_dict: dict):
        """
        Format of loss_dict:
        {
            "default(loss_module_name)": {
                "loss_name": {
                    "loss": torch.Tensor(...),
                    "weight": torch.Tensor(...)
                }
            }
        }
        """
        self.loss_values["total"] = 0
        # update loss value
        for loss_info in loss_dict.values():
            for loss_name, loss_data in loss_info.items():
                self.loss_values[loss_name] = loss_data["loss"]
                self.weight_values[loss_name] = loss_data.get("weight", 1.0)
        # compute total sum of losses
        for k in self.loss_values.keys():
            if k != "total":
                self.loss_values["total"] += self.loss_values[k] * self.weight_values[k]
        
        # average the values
        for k in self.loss_values.keys():
            self.avg_loss_list[k].append(float(self.loss_values[k]))

    def average_losses(self):
        for k in self.avg_loss_list.keys():
            sub_loss = torch.tensor(self.avg_loss_list[k])
            avg_sub_loss = torch.mean(sub_loss[~ torch.isnan(sub_loss)])
            self.avg_loss_values[k] = avg_sub_loss
        self.avg_loss_list = defaultdict(list)
        
    
    def get(self, key, default_value=None, is_avg_value=True):
        if is_avg_value:
            return self.avg_loss_values.get(key, default_value)
        else:
            return self.loss_values.get(key, default_value)

if __name__ == "__main__":
    test = LossHandler()
    test.zero_losses()