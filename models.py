import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DualGraphFusionGCN(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        self.conv1_s = GCNConv(input_dim, 64)
        self.conv2_s = GCNConv(64, 128)
        self.conv1_c = GCNConv(input_dim, 64)
        self.conv2_c = GCNConv(64, 128)

        self.attn = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Linear(128, 9)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, sample_data, cluster_data, Q, alpha=1.0):
        xs = F.relu(self.conv1_s(sample_data.x, sample_data.edge_index))
        xs = F.dropout(xs, p=0.4, training=self.training)
        xs = F.relu(self.conv2_s(xs, sample_data.edge_index))

        xc = F.relu(self.conv1_c(cluster_data.x, cluster_data.edge_index))
        xc = F.dropout(xc, p=0.4, training=self.training)
        xc = F.relu(self.conv2_c(xc, cluster_data.edge_index))

        xc_decoded = Q @ xc
        fusion = torch.cat([xs, xc_decoded], dim=1)
        weights = self.attn(fusion)
        fused = weights[:, 0:1] * xs + weights[:, 1:2] * xc_decoded

        class_output = self.classifier(fused)
        reversed_feat = GradientReversalFunction.apply(fused, alpha)
        domain_output = self.domain_discriminator(reversed_feat)
        return class_output, domain_output, fused