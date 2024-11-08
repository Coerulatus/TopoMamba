""" Taken from https://github.com/pyt-team/TopoModelX/tree/main"""

import os
import time
import numpy as np
import math
from typing import Literal

import torch
from torch.nn.parameter import Parameter

import pytorch_lightning as pl
from topomamba.evaluators.evaluator import Evaluator
from topomamba.modules.mlp import MLP

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """Broadcasts `src` to the shape of `other`."""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Add all values from the `src` tensor into `out` at the indices."""
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)

    return out.scatter_add_(dim, index, src)


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Add all values from the `src` tensor into `out` at the indices."""
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Compute the mean value of all values from the `src` tensor into `out`."""
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out

SCATTER_DICT = {"sum": scatter_sum, "mean": scatter_mean, "add": scatter_sum}

def scatter(scatter: str):
    """Return the scatter function."""
    if scatter not in SCATTER_DICT:
        raise ValueError(f"scatter must be string: {list(SCATTER_DICT.keys())}")

    return SCATTER_DICT[scatter]

class MessagePassing(torch.nn.Module):
    """Define message passing.

    This class defines message passing through a single neighborhood N,
    by decomposing it into 2 steps:

    1. 🟥 Create messages going from source cells to target cells through N.
    2. 🟧 Aggregate messages coming from different sources cells onto each target cell.

    This class should not be instantiated directly, but rather inherited
    through subclasses that effectively define a message passing function.

    This class does not have trainable weights, but its subclasses should
    define these weights.

    Parameters
    ----------
    aggr_func : Literal["sum", "mean", "add"], default="sum"
        Aggregation function to use.
    att : bool, default=False
        Whether to use attention.
    initialization : Literal["uniform", "xavier_uniform", "xavier_normal"], default="xavier_uniform"
        Initialization method for the weights of the layer.
    initialization_gain : float, default=1.414
        Gain for the weight initialization.

    References
    ----------
    .. [1] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz, Ramamurthy, Birdal, Dey,
        Mukherjee, Samaga, Livesay, Walters, Rosen, Schaub.
        Topological deep learning: going beyond graph data (2023).
        https://arxiv.org/abs/2206.00606.

    .. [2] Papillon, Sanborn, Hajij, Miolane.
        Architectures of topological deep learning: a survey on topological neural networks (2023).
        https://arxiv.org/abs/2304.10031.
    """

    def __init__(
        self,
        aggr_func: Literal["sum", "mean", "add"] = "sum",
        att: bool = False,
        initialization: Literal[
            "uniform", "xavier_uniform", "xavier_normal"
        ] = "xavier_uniform",
        initialization_gain: float = 1.414,
    ) -> None:
        super().__init__()
        self.aggr_func = aggr_func
        self.att = att
        self.initialization = initialization
        self.initialization_gain = initialization_gain

    def reset_parameters(self):
        r"""Reset learnable parameters.

        Notes
        -----
        This function will be called by subclasses of MessagePassing that have trainable weights.
        """
        match self.initialization:
            case "uniform":
                if self.weight is not None:
                    stdv = 1.0 / math.sqrt(self.weight.size(1))
                    self.weight.data.uniform_(-stdv, stdv)
                if self.att:
                    stdv = 1.0 / math.sqrt(self.att_weight.size(1))
                    self.att_weight.data.uniform_(-stdv, stdv)
            case "xavier_uniform":
                if self.weight is not None:
                    torch.nn.init.xavier_uniform_(
                        self.weight, gain=self.initialization_gain
                    )
                if self.att:
                    torch.nn.init.xavier_uniform_(
                        self.att_weight.view(-1, 1), gain=self.initialization_gain
                    )
            case "xavier_normal":
                if self.weight is not None:
                    torch.nn.init.xavier_normal_(
                        self.weight, gain=self.initialization_gain
                    )
                if self.att:
                    torch.nn.init.xavier_normal_(
                        self.att_weight.view(-1, 1), gain=self.initialization_gain
                    )
            case _:
                raise ValueError(
                    f"Initialization {self.initialization} not recognized."
                )

    def message(self, x_source, x_target=None):
        """Construct message from source cells to target cells.

        🟥 This provides a default message function to the message passing scheme.

        Alternatively, users can subclass MessagePassing and overwrite
        the message method in order to replace it with their own message mechanism.

        Parameters
        ----------
        x_source : Tensor, shape = (..., n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_target : Tensor, shape = (..., n_target_cells, in_channels)
            Input features on target cells.
            Assumes that all target cells have the same rank s.
            Optional. If not provided, x_target is assumed to be x_source,
            i.e. source cells send messages to themselves.

        Returns
        -------
        torch.Tensor, shape = (..., n_source_cells, in_channels)
            Messages on source cells.
        """
        return x_source

    def attention(self, x_source, x_target=None):
        """Compute attention weights for messages.

        This provides a default attention function to the message-passing scheme.

        Alternatively, users can subclass MessagePassing and overwrite
        the attention method in order to replace it with their own attention mechanism.

        The implementation follows [1]_.

        Parameters
        ----------
        x_source : torch.Tensor, shape = (n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_target : torch.Tensor, shape = (n_target_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.

        Returns
        -------
        torch.Tensor, shape = (n_messages, 1)
            Attention weights: one scalar per message between a source and a target cell.
        """
        x_source_per_message = x_source[self.source_index_j]
        x_target_per_message = (
            x_source[self.target_index_i]
            if x_target is None
            else x_target[self.target_index_i]
        )

        x_source_target_per_message = torch.cat(
            [x_source_per_message, x_target_per_message], dim=1
        )

        return torch.nn.functional.elu(
            torch.matmul(x_source_target_per_message, self.att_weight)
        )

    def aggregate(self, x_message):
        """Aggregate messages on each target cell.

        A target cell receives messages from several source cells.
        This function aggregates these messages into a single output
        feature per target cell.

        🟧 This function corresponds to the within-neighborhood aggregation
        defined in [1]_ and [2]_.

        Parameters
        ----------
        x_message : torch.Tensor, shape = (..., n_messages, out_channels)
            Features associated with each message.
            One message is sent from a source cell to a target cell.

        Returns
        -------
        Tensor, shape = (...,  n_target_cells, out_channels)
            Output features on target cells.
            Each target cell aggregates messages from several source cells.
            Assumes that all target cells have the same rank s.
        """
        aggr = scatter(self.aggr_func)
        return aggr(x_message, self.target_index_i, 0)

    def forward(self, x_source, neighborhood, x_target=None):
        r"""Forward pass.

        This implements message passing for a given neighborhood:

        - from source cells with input features `x_source`,
        - via `neighborhood` defining where messages can pass,
        - to target cells with input features `x_target`.

        In practice, this will update the features on the target cells.

        If not provided, x_target is assumed to be x_source,
        i.e. source cells send messages to themselves.

        The message passing is decomposed into two steps:

        1. 🟥 Message: A message :math:`m_{y \rightarrow x}^{\left(r \rightarrow s\right)}`
        travels from a source cell :math:`y` of rank r to a target cell :math:`x` of rank s
        through a neighborhood of :math:`x`, denoted :math:`\mathcal{N} (x)`,
        via the message function :math:`M_\mathcal{N}`:

        .. math::
            m_{y \rightarrow x}^{\left(r \rightarrow s\right)}
                = M_{\mathcal{N}}\left(\mathbf{h}_x^{(s)}, \mathbf{h}_y^{(r)}, \Theta \right),

        where:

        - :math:`\mathbf{h}_y^{(r)}` are input features on the source cells, called `x_source`,
        - :math:`\mathbf{h}_x^{(s)}` are input features on the target cells, called `x_target`,
        - :math:`\Theta` are optional parameters (weights) of the message passing function.

        Optionally, attention can be applied to the message, such that:

        .. math::
            m_{y \rightarrow x}^{\left(r \rightarrow s\right)}
                \leftarrow att(\mathbf{h}_y^{(r)}, \mathbf{h}_x^{(s)}) . m_{y \rightarrow x}^{\left(r \rightarrow s\right)}

        2. 🟧 Aggregation: Messages are aggregated across source cells :math:`y` belonging to the
        neighborhood :math:`\mathcal{N}(x)`:

        .. math::
            m_x^{\left(r \rightarrow s\right)}
                = \text{AGG}_{y \in \mathcal{N}(x)} m_{y \rightarrow x}^{\left(r\rightarrow s\right)},

        resulting in the within-neighborhood aggregated message :math:`m_x^{\left(r \rightarrow s\right)}`.

        Details can be found in [1]_ and [2]_.

        Parameters
        ----------
        x_source : Tensor, shape = (..., n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape = (n_target_cells, n_source_cells)
            Neighborhood matrix.
        x_target : Tensor, shape = (..., n_target_cells, in_channels)
            Input features on target cells.
            Assumes that all target cells have the same rank s.
            Optional. If not provided, x_target is assumed to be x_source,
            i.e. source cells send messages to themselves.

        Returns
        -------
        torch.Tensor, shape = (..., n_target_cells, out_channels)
            Output features on target cells.
            Assumes that all target cells have the same rank s.
        """
        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()
        neighborhood_values = neighborhood.values()

        x_message = self.message(x_source=x_source, x_target=x_target)
        x_message = x_message.index_select(-2, self.source_index_j)

        if self.att:
            attention_values = self.attention(x_source=x_source, x_target=x_target)
            neighborhood_values = torch.multiply(neighborhood_values, attention_values)

        x_message = neighborhood_values.view(-1, 1) * x_message
        return self.aggregate(x_message)

class Conv(MessagePassing):
    """Message passing: steps 1, 2, and 3.

    Builds the message passing route given by one neighborhood matrix.
    Includes an option for an x-specific update function.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    aggr_norm : bool, default=False
        Whether to normalize the aggregated message by the neighborhood size.
    update_func : {"relu", "sigmoid"}, optional
        Update method to apply to message.
    att : bool, default=False
        Whether to use attention.
    initialization : {"xavier_uniform", "xavier_normal"}, default="xavier_uniform"
        Initialization method.
    initialization_gain : float, default=1.414
        Initialization gain.
    with_linear_transform : bool, default=True
        Whether to apply a learnable linear transform.
        NB: if `False` in_channels has to be equal to out_channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        aggr_norm: bool = False,
        update_func: Literal["relu", "sigmoid", None] = None,
        att: bool = False,
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
        initialization_gain: float = 1.414,
        with_linear_transform: bool = True,
    ) -> None:
        super().__init__(
            att=att,
            initialization=initialization,
            initialization_gain=initialization_gain,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr_norm = aggr_norm
        self.update_func = update_func

        self.weight = (
            Parameter(torch.Tensor(self.in_channels, self.out_channels))
            if with_linear_transform
            else None
        )

        if not with_linear_transform and in_channels != out_channels:
            raise ValueError(
                "With `linear_trainsform=False`, in_channels has to be equal to out_channels"
            )
        if self.att:
            self.att_weight = Parameter(
                torch.Tensor(
                    2 * self.in_channels,
                )
            )

        self.reset_parameters()

    def update(self, x_message_on_target) -> torch.Tensor:
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x_message_on_target : torch.Tensor, shape = (n_target_cells, out_channels)
            Output features on target cells.

        Returns
        -------
        torch.Tensor, shape = (n_target_cells, out_channels)
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x_message_on_target)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x_message_on_target)
        return x_message_on_target

    def forward(self, x_source, neighborhood, x_target=None) -> torch.Tensor:
        """Forward pass.

        This implements message passing:
        - from source cells with input features `x_source`,
        - via `neighborhood` defining where messages can pass,
        - to target cells with input features `x_target`.

        In practice, this will update the features on the target cells.

        If not provided, x_target is assumed to be x_source,
        i.e. source cells send messages to themselves.

        Parameters
        ----------
        x_source : Tensor, shape = (..., n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape = (n_target_cells, n_source_cells)
            Neighborhood matrix.
        x_target : Tensor, shape = (..., n_target_cells, in_channels)
            Input features on target cells.
            Assumes that all target cells have the same rank s.
            Optional. If not provided, x_target is assumed to be x_source,
            i.e. source cells send messages to themselves.

        Returns
        -------
        torch.Tensor, shape = (..., n_target_cells, out_channels)
            Output features on target cells.
            Assumes that all target cells have the same rank s.
        """
        if self.att:
            neighborhood = neighborhood.coalesce()
            self.target_index_i, self.source_index_j = neighborhood.indices()
            attention_values = self.attention(x_source, x_target)
            neighborhood = torch.sparse_coo_tensor(
                indices=neighborhood.indices(),
                values=attention_values * neighborhood.values(),
                size=neighborhood.shape,
            )
        if self.weight is not None:
            x_message = torch.mm(x_source, self.weight)
        else:
            x_message = x_source
        x_message_on_target = torch.mm(neighborhood, x_message)

        if self.aggr_norm:
            neighborhood_size = torch.sum(neighborhood.to_dense(), dim=1)
            x_message_on_target = torch.einsum(
                "i,ij->ij", 1 / neighborhood_size, x_message_on_target
            )

        return self.update(x_message_on_target)
    
class SCN2Layer(torch.nn.Module):
    """Layer of a Simplex Convolutional Network (SCN).

    Implementation of the SCN layer proposed in [1]_ for a simplicial complex of
    rank 2, that is for 0-cells (nodes), 1-cells (edges) and 2-cells (faces) only.

    This layer corresponds to the rightmost tensor diagram labeled Yang22c in
    Figure 11 of [PSHM23]_.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on nodes (0-cells).
    in_channels_1 : int
        Dimension of input features on edges (1-cells).
    in_channels_2 : int
        Dimension of input features on faces (2-cells).

    See Also
    --------
    topomodelx.nn.simplicial.sccn_layer.SCCNLayer : SCCN layer
        Simplicial Complex Convolutional Network (SCCN) layer proposed in [1]_.
        The difference between SCCN and SCN is that:
        - SCN passes messages between cells of the same rank,
        - SCCN passes messages between cells of the same ranks, one rank above
        and one rank below.

    Notes
    -----
    This architecture is proposed for simplicial complex classification.

    References
    ----------
    .. [1] Yang, Sala and Bogdan.
        Efficient representation learning for higher-order data with simplicial complexes (2022).
        https://proceedings.mlr.press/v198/yang22a.html.
    .. [2] Papillon, Sanborn, Hajij, Miolane.
        Equations of topological neural networks (2023).
        https://github.com/awesome-tnns/awesome-tnns/
    .. [3] Papillon, Sanborn, Hajij, Miolane.
        Architectures of topological deep learning: a survey on topological neural networks (2023).
        https://arxiv.org/abs/2304.10031.
    """

    def __init__(self, in_channels_0, in_channels_1, in_channels_2) -> None:
        super().__init__()
        self.conv_0_to_0 = Conv(in_channels=in_channels_0, out_channels=in_channels_0)
        self.conv_1_to_1 = Conv(in_channels=in_channels_1, out_channels=in_channels_1)
        self.conv_2_to_2 = Conv(in_channels=in_channels_2, out_channels=in_channels_2)

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        self.conv_0_to_0.reset_parameters()
        self.conv_1_to_1.reset_parameters()
        self.conv_2_to_2.reset_parameters()

    def forward(self, x_0, x_1, x_2, laplacian_0, laplacian_1, laplacian_2):
        r"""Forward pass (see [2]_ and [3]_).

        .. math::
            \begin{align*}
            &🟥 \quad m^{(r \rightarrow r)}\_{y \rightarrow x}  = (2I + H_r)\_{{xy}} \cdot h_{y}^{t,(1)}\cdot \Theta^t\\
            &🟧 \quad m_x^{(1 \rightarrow 1)}  = \sum_{y \in (\mathcal{L}\_\downarrow+\mathcal{L}\_\uparrow)(x)} m_{y \rightarrow x}^{(1 \rightarrow 1)}\\
            &🟩 \quad m_x^{(1)}  = m^{(1 \rightarrow 1)}_x\\
            &🟦 \quad h_x^{t+1,(1)} = \sigma(m_{x}^{(1)})
            \end{align*}

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, node_features)
            Input features on the nodes of the simplicial complex.
        x_1 : torch.Tensor, shape = (n_edges, edge_features)
            Input features on the edges of the simplicial complex.
        x_2 : torch.Tensor, shape = (n_faces, face_features)
            Input features on the faces of the simplicial complex.
        laplacian_0 : torch.sparse, shape = (n_nodes, n_nodes)
            Normalized Hodge Laplacian matrix = L_upper + L_lower.
        laplacian_1 : torch.sparse, shape = (n_edges, n_edges)
            Normalized Hodge Laplacian matrix.
        laplacian_2 : torch.sparse, shape = (n_faces, n_faces)
            Normalized Hodge Laplacian matrix.

        Returns
        -------
        torch.Tensor, shape = (n_nodes, channels)
            Output features on the nodes of the simplicial complex.
        """
        x_0 = self.conv_0_to_0(x_0, laplacian_0)
        x_0 = torch.nn.functional.relu(x_0)
        x_1 = self.conv_1_to_1(x_1, laplacian_1)
        x_1 = torch.nn.functional.relu(x_1)
        x_2 = self.conv_2_to_2(x_2, laplacian_2)
        x_2 = torch.nn.functional.relu(x_2)
        return x_0, x_1, x_2
    
class SCN2(torch.nn.Module):
    """Simplex Convolutional Network Implementation for binary node classification.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on nodes.
    in_channels_1 : int
        Dimension of input features on edges.
    in_channels_2 : int
        Dimension of input features on faces.
    n_layers : int
        Amount of message passing layers.

    """

    def __init__(self, in_channels_0, in_channels_1, in_channels_2, n_layers=2):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            SCN2Layer(
                in_channels_0=in_channels_0,
                in_channels_1=in_channels_1,
                in_channels_2=in_channels_2,
            )
            for _ in range(n_layers)
        )

    def forward(self, x_0, x_1, x_2, laplacian_0, laplacian_1, laplacian_2):
        """Forward computation.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, channels)
            Node features.

        x_1 : torch.Tensor, shape = (n_edges, channels)
            Edge features.

        x_2 : torch.Tensor, shape = (n_faces, channels)
            Face features.

        Returns
        -------
        x_0 : torch.Tensor, shape = (n_nodes, channels)
            Final node hidden states.

        x_1 : torch.Tensor, shape = (n_nodes, channels)
            Final edge hidden states.

        x_2 : torch.Tensor, shape = (n_nodes, channels)
            Final face hidden states.

        """
        for layer in self.layers:
            x_0, x_1, x_2 = layer(x_0, x_1, x_2, laplacian_0, laplacian_1, laplacian_2)

        return x_0, x_1, x_2

class SCCNNLayer(torch.nn.Module):
    r"""Layer of a Simplicial Complex Convolutional Neural Network.

    Parameters
    ----------
    in_channels : tuple of int
        Dimensions of input features on nodes, edges, and faces.
    out_channels : tuple of int
        Dimensions of output features on nodes, edges, and faces.
    conv_order : int
        Convolution order of the simplicial filters.
    sc_order : int
        SC order.
    aggr_norm : bool, optional
        Whether to normalize the aggregated message by the neighborhood size (default: False).
    update_func : str, optional
        Activation function used in aggregation layers (default: None).
    initialization : str, optional
        Initialization method for the weights (default: "xavier_normal").
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_order,
        sc_order = 3,
        aggr_norm: bool = False,
        update_func=None,
        initialization: str = "xavier_normal",
    ) -> None:
        super().__init__()

        in_channels_0, in_channels_1, in_channels_2 = in_channels
        out_channels_0, out_channels_1, out_channels_2 = out_channels

        self.in_channels_0 = in_channels_0
        self.in_channels_1 = in_channels_1
        self.in_channels_2 = in_channels_2
        self.out_channels_0 = out_channels_0
        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2

        self.conv_order = conv_order
        self.sc_order = sc_order

        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization

        assert initialization in ["xavier_uniform", "xavier_normal"]
        assert self.conv_order > 0

        self.weight_0 = Parameter(
            torch.Tensor(
                self.in_channels_0,
                self.out_channels_0,
                1 + conv_order + 1 + conv_order,
            )
        )

        self.weight_1 = Parameter(
            torch.Tensor(
                self.in_channels_1,
                self.out_channels_1,
                6 * conv_order + 3,
            )
        )

        # determine the third dimensions of the weights
        # because when SC order is larger than 2, there are lower and upper
        # parts for L_2; otherwise, L_2 contains only the lower part

        if sc_order > 2:
            self.weight_2 = Parameter(
                torch.Tensor(
                    self.in_channels_2,
                    self.out_channels_2,
                    4 * conv_order
                    + 2,  # in the future for arbitrary sc_order we should have this 6*conv_order + 3,
                )
            )

        elif sc_order == 2:
            self.weight_2 = Parameter(
                torch.Tensor(
                    self.in_channels_2,
                    self.out_channels_2,
                    4 * conv_order + 2,
                )
            )

        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.
        """
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight_0, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight_1, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight_2, gain=gain)
        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.weight_0, gain=gain)
            torch.nn.init.xavier_normal_(self.weight_1, gain=gain)
            torch.nn.init.xavier_normal_(self.weight_2, gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def aggr_norm_func(self, conv_operator, x):
        r"""Perform aggregation normalization.

        Parameters
        ----------
        conv_operator : torch.sparse
            Convolution operator.
        x : torch.Tensor
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Normalized feature tensor.
        """
        neighborhood_size = torch.sum(conv_operator.to_dense(), dim=1)
        neighborhood_size_inv = 1 / neighborhood_size
        neighborhood_size_inv[~(torch.isfinite(neighborhood_size_inv))] = 0

        x = torch.einsum("i,ij->ij ", neighborhood_size_inv, x)
        x[~torch.isfinite(x)] = 0
        return x

    def update(self, x):
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Updated tensor.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x)
        return None

    def chebyshev_conv(self, conv_operator, conv_order, x):
        r"""Perform Chebyshev convolution.

        Parameters
        ----------
        conv_operator : torch.sparse
            Convolution operator.
        conv_order : int
            Order of the convolution.
        x : torch.Tensor
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        num_simplices, num_channels = x.shape
        X = torch.empty(size=(num_simplices, num_channels, conv_order)).to(
            x.device
        )

        if self.aggr_norm:
            X[:, :, 0] = torch.mm(conv_operator, x)
            X[:, :, 0] = self.aggr_norm_func(conv_operator, X[:, :, 0])
            for k in range(1, conv_order):
                X[:, :, k] = torch.mm(conv_operator, X[:, :, k - 1])
                X[:, :, k] = self.aggr_norm_func(conv_operator, X[:, :, k])
        else:
            X[:, :, 0] = torch.mm(conv_operator, x)
            for k in range(1, conv_order):
                X[:, :, k] = torch.mm(conv_operator, X[:, :, k - 1])
        return X

    def forward(self, x_all, laplacian_all, incidence_all):
        r"""Forward computation.

        Parameters
        ----------
        x_all : tuple of tensors
            Tuple of input feature tensors (node, edge, face).
        laplacian_all : tuple of tensors
            Tuple of Laplacian tensors (graph laplacian L0, down edge laplacian L1_d, upper edge laplacian L1_u, face laplacian L2).
        incidence_all : tuple of tensors
            Tuple of order 1 and 2 incidence matrices.

        Returns
        -------
        torch.Tensor
            Output tensor for each 0-cell.
        torch.Tensor
            Output tensor for each 1-cell.
        torch.Tensor
            Output tensor for each 2-cell.
        """
        x_0, x_1, x_2 = x_all

        if self.sc_order == 2:
            laplacian_0, laplacian_down_1, laplacian_up_1, laplacian_2 = (
                laplacian_all
            )
        elif self.sc_order > 2:
            (
                laplacian_0,
                laplacian_down_1,
                laplacian_up_1,
                laplacian_down_2,
                laplacian_up_2,
            ) = laplacian_all
            
        b1, b2 = incidence_all
        """
        Convolution in the node space
        """
        x_0_laplacian = self.chebyshev_conv(laplacian_0, self.conv_order, x_0)
        x_0_to_0 = torch.cat([x_0.unsqueeze(2), x_0_laplacian], dim=2)
        # -------------------
        x_1_to_0_upper = torch.mm(b1, x_1)
        x_1_to_0_laplacian = self.chebyshev_conv(
            laplacian_0, self.conv_order, x_1_to_0_upper
        )
        x_1_to_0 = torch.cat(
            [x_1_to_0_upper.unsqueeze(2), x_1_to_0_laplacian], dim=2
        )
        # -------------------

        x_0_all = torch.cat((x_0_to_0, x_1_to_0), 2)

        # -------------------
        """
        Convolution in the edge space
        """
        x_1_down = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_1)
        x_1_up = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_1)
        x_1_to_1 = torch.cat((x_1.unsqueeze(2), x_1_down, x_1_up), 2)

        # -------------------
        # Lower projection
        x_0_1_lower = torch.mm(b1.T, x_0)

        # Calculate lowwer chebyshev_conv
        x_0_1_down = self.chebyshev_conv(
            laplacian_down_1, self.conv_order, x_0_1_lower
        )

        # Calculate upper chebyshev_conv (Note: in case of signed incidence should be always zero)
        x_0_1_up = self.chebyshev_conv(
            laplacian_up_1, self.conv_order, x_0_1_lower
        )

        # Concatenate output of filters
        x_0_to_1 = torch.cat(
            [x_0_1_lower.unsqueeze(2), x_0_1_down, x_0_1_up], dim=2
        )
        x_2_1_upper = torch.mm(b2, x_2)

        # Calculate lowwer chebyshev_conv (Note: In case of signed incidence should be always zero)
        x_2_1_down = self.chebyshev_conv(
            laplacian_down_1, self.conv_order, x_2_1_upper
        )

        # Calculate upper chebyshev_conv
        x_2_1_up = self.chebyshev_conv(
            laplacian_up_1, self.conv_order, x_2_1_upper
        )

        x_2_to_1 = torch.cat(
            [x_2_1_upper.unsqueeze(2), x_2_1_down, x_2_1_up], dim=2
        )

        # -------------------
        x_1_all = torch.cat((x_0_to_1, x_1_to_1, x_2_to_1), 2)
        """Convolution in the face (triangle) space, depending on the SC order,
        the exact form maybe a little different."""
        x_2_down = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_2)
        x_2_up = self.chebyshev_conv(laplacian_up_2, self.conv_order, x_2)
        x_2_to_2 = torch.cat((x_2.unsqueeze(2), x_2_down, x_2_up), 2)

        x_1_2_lower = torch.mm(b2.T, x_1)
        x_1_2_down = self.chebyshev_conv(
            laplacian_down_2, self.conv_order, x_1_2_lower
        )
        x_1_2_down = self.chebyshev_conv(
            laplacian_up_2, self.conv_order, x_1_2_lower
        )

        x_1_to_2 = torch.cat(
            [x_1_2_lower.unsqueeze(2), x_1_2_down, x_1_2_down], dim=2
        )

        x_2_all = torch.cat([x_1_to_2, x_2_to_2], dim=2)

        y_0 = torch.einsum("nik,iok->no", x_0_all, self.weight_0)

        if self.update_func is None:
            return y_0

        return self.update(y_0)
     
class Wrapper(pl.LightningModule):
    def __init__(self,
                 d_input,
                 d_out,
                 d_hidden = 256,
                 n_layers = 2,
                 task_level = "node",
                 task = "classification",
                 pooling_type = "sum",
                 loss = torch.nn.CrossEntropyLoss(),
                 lr = 0.01,
                 input_dropout = 0.1,
                 readout_dropout = 0.1,
                 device = "cuda",
                 log = True,
                 batch_size = 1,
                 model = 'scn',
                 save_results = False,
                 time_start = time.time()):
        super(Wrapper, self).__init__()
        self.loss = loss
        self.lr = lr
        
        self.input_transform = torch.nn.Sequential(torch.nn.Linear(d_input, d_hidden, device=device),
                                                    torch.nn.Dropout(input_dropout),
                                                    torch.nn.ReLU(),)
        self.layer_norm = torch.nn.LayerNorm(d_hidden, device=device)
        self.model = model
        if model == 'scn':
            self.layers = torch.nn.ModuleList(
                SCN2Layer(
                    in_channels_0=d_hidden,
                    in_channels_1=d_hidden,
                    in_channels_2=d_hidden,
                )
                for _ in range(n_layers)
            )
        elif model == 'sccnn':
            self.layers = torch.nn.ModuleList(
                SCCNNLayer(
                    in_channels=(d_hidden, d_hidden, d_hidden),
                    out_channels=(d_hidden, d_hidden, d_hidden),
                    conv_order=2,
                    sc_order=3,
                    aggr_norm=True,
                    update_func=None,
                    initialization="xavier_normal",
                )
                for _ in range(n_layers)
            )
        self.layers.to(device)
        self.readout = MLP(in_channels=d_hidden,
                           hidden_channels=d_hidden,
                           out_channels=d_out,
                           num_layers=2,
                           dropout=readout_dropout,
                           Normalization='bn',
                           InputNorm=False)
        self.task_level = task_level
        self.task = task
        self.pooling_type = pooling_type
        self.evaluator = Evaluator(task=task)
        self.train_results = {"logits": [], "labels": []}
        self.val_results = {"logits": [], "labels": []}
        self.test_results = {"logits": [], "labels": []}
        self.do_log = log
        self.batch_size = batch_size
        self.save_hyperparameters(ignore=['loss'])
        self.train_times = []
        self.train_loss = []
        # self.train_acc = []
        self.train_iter = []
        self.train_epoch = []
        self.val_times = []
        self.val_acc = []
        self.n_epoch = 1
        self.time = time_start
        self.save_results = save_results
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.params = sum([np.prod(p.size()) for p in model_parameters])
        
    def forward(self, batch):
        laplacian_0 = self.normalize_matrix(batch.laplacian_0)
        incidence_1 = batch.incidence_1
        incidence_2 = batch.incidence_2
        if self.model == 'scn':
            laplacian_1 = self.normalize_matrix(batch.laplacian_1)
            laplacian_2 = self.normalize_matrix(batch.laplacian_2)
        elif self.model == 'sccnn':
            down_laplacian_1 = self.normalize_matrix(batch.down_laplacian_1)
            up_laplacian_1 = self.normalize_matrix(batch.up_laplacian_1)
            down_laplacian_2 = self.normalize_matrix(batch.down_laplacian_2)
            if hasattr(batch, 'up_laplacian_2'):
                up_laplacian_2 = self.normalize_matrix(batch.up_laplacian_2)
        x = self.input_transform(batch.x)
        for layer in self.layers:
            x_0, x_1, x_2 = self.get_features(x, incidence_1, incidence_2)
            if self.model == 'scn':
                x, _, _ = layer(x_0, x_1, x_2, laplacian_0, laplacian_1, laplacian_2)
            elif self.model == 'sccnn':
                x = layer((x_0, x_1, x_2), (laplacian_0, down_laplacian_1, up_laplacian_1, down_laplacian_2, up_laplacian_2), (incidence_1, incidence_2))
        if self.task_level == "graph":
            if self.pooling_type == "sum":
                x = torch.sum(x, dim=0)
            elif self.pooling_type == "mean":
                x = torch.mean(x, dim=0)
            elif self.pooling_type == "max":
                x = torch.max(x, dim=0)
            else:
                raise NotImplementedError
        x = self.readout(x)
        return x
    
    def balance_loss(self, y):
        if max(y) == 1:
            num_ones = torch.sum(y)
            num_zeros = len(y) - num_ones
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1/num_zeros, 1/num_ones]).float().to(y.device))
        else:
            raise NotImplementedError("Balance loss only implemented for binary classification")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def get_features(self, x, incidence_1, incidence_2):
        x_0 = x.view(x.shape[0], -1)
        x_1 = torch.sparse.mm(incidence_1.T, x_0)
        x_2 = torch.sparse.mm(incidence_2.T, x_1)
        return x_0, x_1, x_2
    
    def normalize_matrix(self, matrix):
        r"""Normalize the input matrix.

        The normalization is performed using the diagonal matrix of the inverse square root of the sum of the absolute values of the rows.

        Parameters
        ----------
        matrix : torch.sparse.FloatTensor
            Input matrix to be normalized.

        Returns
        -------
        torch.sparse.FloatTensor
            Normalized matrix.
        """
        matrix_ = matrix.to_dense()
        n, _ = matrix_.shape
        abs_matrix = abs(matrix_)
        diag_sum = abs_matrix.sum(axis=1)

        # Handle division by zero
        idxs = torch.where(diag_sum != 0)
        diag_sum[idxs] = 1.0 / torch.sqrt(diag_sum[idxs])

        diag_indices = torch.stack([torch.arange(n), torch.arange(n)])
        diag_matrix = torch.sparse_coo_tensor(
            diag_indices, diag_sum, matrix_.shape, device=matrix.device
        ).coalesce()
        normalized_matrix = diag_matrix @ (matrix @ diag_matrix)
        return normalized_matrix

    def training_step(self, batch, batch_idx):
        if self.task == "classification":
            mask = torch.tensor([i for i in range(batch.batch_size)])
        else:
            mask = torch.tensor([i for i in range(batch.y.shape[0])])
        y_true = batch.y[mask]
        mask = mask.to('cpu')
        y_pred = self.forward(batch)
        mask = mask.to(batch.x.device)
        y_pred = y_pred[mask]
        if self.task == "classification":
            loss = self.loss(y_pred, torch.nn.functional.one_hot(y_true, num_classes=y_pred.shape[1]).float())
        else:
            loss = self.loss(y_pred.squeeze(), y_true)
        if self.do_log:
            if self.batch_size == 'full':
                batch_size = len(y_true)
            else:
                batch_size = self.batch_size
            self.log("train_loss", loss, batch_size=batch_size)
        self.train_results["logits"].append(y_pred.detach())
        self.train_results["labels"].append(y_true)
        if hasattr(self, 'time_start_train'):
            self.train_iter.append(time.time()-self.time_start_train)
        self.time_start_train = time.time()
        self.train_times.append(time.time()-self.time)
        self.train_loss.append(loss.cpu().detach().numpy())
        return loss
    
    def on_train_epoch_end(self):
        if len(self.train_results["logits"][0].shape) == 0:
            y_pred = torch.stack(self.train_results["logits"],dim=0)
            y_true = torch.stack(self.train_results["labels"],dim=0)
        else:
            y_pred = torch.cat(self.train_results["logits"],dim=0)
            y_true = torch.cat(self.train_results["labels"],dim=0)
        eval = self.evaluator.eval({"logits": y_pred.squeeze(), "labels": y_true})
        for key in eval:
            if self.do_log:
                if self.batch_size == 'full':
                    batch_size = len(y_true)
                else:
                    batch_size = self.batch_size
                self.log("train_"+key, eval[key], batch_size=batch_size)
                self.log("a_train_iter", np.mean(self.train_iter))
                self.log("a_train_iter_std", np.std(self.train_iter))
                self.log("a_train_epoch", np.mean(self.train_epoch))
                self.log("a_train_epoch_std", np.std(self.train_epoch))  
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
                    torch.cuda.reset_peak_memory_stats()
                    self.log("train_gpu_mem", gpu_memory, batch_size=batch_size)
                    self.log("_n_params", self.params)
        self.train_results = {"logits": [], "labels": []}
        if hasattr(self, 't_epoch'):
            self.train_epoch.append(time.time()-self.t_epoch)
        self.t_epoch = time.time()

    def validation_step(self, batch, batch_idx):
        if self.task == "classification":
            mask = torch.tensor([i for i in range(batch.batch_size)])
        else:
            mask = torch.tensor([i for i in range(batch.y.shape[0])])
        y_true = batch.y[mask]
        mask = mask.to('cpu')
        y_pred = self.forward(batch)
        mask = mask.to(batch.x.device)
        y_pred = y_pred[mask]
        if self.task == "classification":
            loss = self.loss(y_pred, torch.nn.functional.one_hot(y_true, num_classes=y_pred.shape[1]).float())
        else:
            loss = self.loss(y_pred.squeeze(), y_true)
        if self.do_log:
            if self.batch_size == 'full':
                batch_size = len(y_true)
            else:
                batch_size = self.batch_size
            self.log("valid_loss", loss, batch_size=batch_size)
        self.val_results["logits"].append(y_pred.detach())
        self.val_results["labels"].append(y_true)
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_results["logits"][0].shape) == 0:
            y_pred = torch.stack(self.val_results["logits"],dim=0)
            y_true = torch.stack(self.val_results["labels"],dim=0)
        else:
            y_pred = torch.cat(self.val_results["logits"],dim=0)
            y_true = torch.cat(self.val_results["labels"],dim=0)
        eval = self.evaluator.eval({"logits": y_pred.squeeze(), "labels": y_true})
        for key in eval:
            if self.batch_size == 'full':
                batch_size = len(y_true)
            else:
                batch_size = self.batch_size
            if key == 'acc':
                self.log("valid_acc", eval["acc"], prog_bar=True, batch_size=batch_size)
            elif self.do_log:
                self.log("valid_"+key, eval[key], batch_size=batch_size)
            self.val_times.append(time.time()-self.time)
            self.val_acc.append(eval["acc"])
        self.val_results = {"logits": [], "labels": []}
        
    def on_train_end(self) -> None:
        if self.save_results:
            idx = 0
            # check if the file already exists
            while os.path.exists(f'./results/train_loss{idx}.npy'):
                idx += 1
            # save train_loss and train_time with numpy
            with open(f'./results/train_loss{idx}.npy', 'wb') as f:
                np.save(f, self.train_loss)
            with open(f'./results/train_time{idx}.npy', 'wb') as f:
                np.save(f, self.train_times)
            with open(f'./results/val_acc{idx}.npy', 'wb') as f:
                np.save(f, self.val_acc)
            with open(f'./results/val_time{idx}.npy', 'wb') as f:
                np.save(f, self.val_times)
        return super().on_train_end()

    def test_step(self, batch, batch_idx):
        if self.task == "classification":
            mask = torch.tensor([i for i in range(batch.batch_size)])
        else:
            mask = torch.tensor([i for i in range(batch.y.shape[0])])
        y_true = batch.y[mask]
        mask = mask.to('cpu')
        y_pred = self.forward(batch)
        mask = mask.to(batch.x.device)
        y_pred = y_pred[mask]
        self.test_results["logits"].append(y_pred.detach())
        self.test_results["labels"].append(y_true)
        
    def on_test_epoch_end(self):
        if len(self.test_results["logits"][0].shape) == 0:
            y_pred = torch.stack(self.test_results["logits"],dim=0)
            y_true = torch.stack(self.test_results["labels"],dim=0)
        else:
            y_pred = torch.cat(self.test_results["logits"],dim=0)
            y_true = torch.cat(self.test_results["labels"],dim=0)
        eval = self.evaluator.eval({"logits": y_pred.squeeze(), "labels": y_true})
        for key in eval:
            if self.do_log:
                if self.batch_size == 'full':
                    batch_size = len(y_true)
                else:
                    batch_size = self.batch_size
                self.log("test_"+key, eval[key], batch_size=batch_size)
        self.test_results = {"logits": [], "labels": []}
        return eval