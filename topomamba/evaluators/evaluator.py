import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


### Evaluator for graph classification
class Evaluator:
    def __init__(self, task="classification", metrics=[]):
        if task == "classification":
            accepted_metrics = ["acc", "pre", "rec", "f1","rocauc"]
        elif task == "regression":
            accepted_metrics = ["mae"]
            
        if len(metrics) == 0:
            if task == "classification":
                metrics = ["acc", "pre", "rec", "f1", "rocauc"]
            else:
                metrics = ["mae"]
        for m in metrics:
            if m not in accepted_metrics:
                raise ValueError(
                    f"Metric {m} for {task} is not valid. Choose betweeen {accepted_metrics}."
                )
        self.metrics = metrics
        self.task = task

    @property
    def expected_input_format(self):
        input_format = f"Expected input format is a numpy.array or a torch.Tensor of shape (samples, n_classes)."
        return input_format

    def _parse_and_check_input(self, input_dict):
        if not "labels" in input_dict:
            raise RuntimeError("Missing key of y_true")
        if not "logits" in input_dict:
            raise RuntimeError("Missing key of y_pred")

        y_true, y_logits = input_dict["labels"], input_dict["logits"]
        if self.task == "classification":
            y_pred = y_logits.argmax(dim=-1)
        else:
            y_pred = y_logits
        """
            y_true: numpy ndarray or torch tensor of shape (num_graphs/n_nodes, n_classes)
            y_pred: numpy ndarray or torch tensor of shape (num_graphs/n_nodes, n_classes)
        """

        # converting to torch.Tensor to numpy on cpu
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        ## check type
        if not isinstance(y_true, np.ndarray):
            raise RuntimeError(
                "Arguments to Evaluator need to be either numpy ndarray or torch tensor"
            )
        if not y_true.shape == y_pred.shape:
            raise RuntimeError("Shape of y_true and y_pred must be the same")

        return y_true, y_pred, y_logits.to('cpu')

    def eval(self, input_dict):
        results = {}
        y_true, y_pred, y_logits = self._parse_and_check_input(input_dict)
        # Correct it later to allow roc_auc_score
        res_true = y_true  # np.argmax(y_true, axis=1)
        res_pred = y_pred  # np.argmax(y_pred, axis=1)
        if self.task == "classification":
            n_classes = input_dict['logits'].shape[1]
        
        for metric in self.metrics:
            if metric == "rocauc":
                if n_classes == 2 or n_classes == 1:
                    results["rocauc"] = roc_auc_score(y_true, y_logits[:,1])
                else:
                    # results["rocauc"] = roc_auc_score(y_true, y_logits, multi_class="ovr")
                    pass
            if metric == "acc":
                results["acc"] = accuracy_score(res_true, res_pred)
            elif metric == "pre":
                if n_classes == 2 or n_classes == 1:
                    results["pre"] = precision_score(res_true, res_pred, zero_division=0)
                else:
                    results["pre_micro"] = precision_score(
                        res_true, res_pred, average="micro", zero_division=0
                    )
                    results["pre_macro"] = precision_score(
                        res_true, res_pred, average="macro", zero_division=0
                    )
            elif metric == "rec":
                if n_classes == 2 or n_classes == 1:
                    results["rec"] = recall_score(res_true, res_pred, zero_division=0)
                else:
                    results["rec_micro"] = recall_score(res_true, res_pred, average="micro", zero_division=0)
                    results["rec_macro"] = recall_score(res_true, res_pred, average="macro", zero_division=0)
            elif metric == "f1":
                if n_classes == 2 or n_classes == 1:
                    results["f1"] = f1_score(res_true, res_pred, zero_division=0)
                else:
                    results["f1_micro"] = f1_score(res_true, res_pred, average="micro", zero_division=0)
                    results["f1_macro"] = f1_score(res_true, res_pred, average="macro", zero_division=0)
            elif metric == "mae":
                results["mae"] = np.mean(np.abs(y_true - y_pred))
        return results


if __name__ == "__main__":
    evaluator = Evaluator()
    print(evaluator.expected_input_format)