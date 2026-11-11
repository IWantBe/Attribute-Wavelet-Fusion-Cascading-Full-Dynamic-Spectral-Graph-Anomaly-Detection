import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as dglF
import scipy.special
import sympy
import sklearn.metrics as skmetrics
from torch.nn.parameter import Parameter

import datasets
from .base import BaseModel, get_thres


def is_heterograph(g: dgl.DGLGraph) -> bool:
    return len(g.ntypes) > 1 or len(g.etypes) > 1


class DimensionWiseLinear(nn.Module):

    def __init__(self, d: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = Parameter(torch.empty(d, in_features, out_features))
        self.b = bias
        if self.b:
            self.bias = Parameter(torch.empty(d, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        bound = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)
        if self.b:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        x = (x[..., None, :] @ self.weight).squeeze(-2)
        if self.b:
            x = x + self.bias
        return x


class FeatureAwareFourierMapping(nn.Module):

    def __init__(self, d: int, d_e: int, sigma: float = 1.0):
        super().__init__()
        self.d = d
        self.d_e = d_e
        self.frequencies = nn.Parameter(torch.randn(d, d_e) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = 2 * math.pi * x[:, :, None] * self.frequencies.unsqueeze(0)
        return torch.cat([torch.sin(u), torch.cos(u)], dim=-1)


class DynamicAttributeTransformation(nn.Module):

    def __init__(
            self,
            d: int,
            d_e: int = 16,
            sigma: float = 5.0,
            p_drop: float = 0.1,
            use_bn: bool = True,
    ):
        super().__init__()
        self.fourier_mapping = FeatureAwareFourierMapping(d, d_e, sigma=sigma)
        self.dimension_wise_linear = DimensionWiseLinear(d, d_e * 2, d_e)
        self.bn0 = nn.BatchNorm1d(d) if use_bn else nn.Identity()
        self.linear = nn.Linear(d * d_e, d)
        self.bn1 = nn.BatchNorm1d(d) if use_bn else nn.Identity()
        self.drop = nn.Dropout(p_drop)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def fourier_transform(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fourier_mapping(x)
        h = self.dimension_wise_linear(h)
        h = self.bn0(h)
        h = F.relu(h)
        h = self.drop(h)
        h = h.flatten(start_dim=1)
        h = self.linear(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.drop(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_hat = self.fourier_transform(x)
        gate = torch.sigmoid(self.alpha)
        return gate * z_hat + (1 - gate) * x


def beta_wavelet_coefficients(C: int) -> list:
    thetas = []
    x = sympy.symbols('x')
    for i in range(C + 1):
        f = sympy.poly(
            (x / 2) ** i * (1 - x / 2) ** (C - i) /
            (scipy.special.beta(i + 1, C + 1 - i))
        )
        coeff = f.all_coeffs()
        thetas.append([float(coeff[C - k]) for k in range(C + 1)])
    return thetas


def beta_wavelet_transform(theta: list, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:

    def unnLaplacian(feat, D_invsqrt, graph):
        graph.ndata['h'] = feat * D_invsqrt
        graph.update_all(dglF.copy_u('h', 'm'), dglF.sum('m', 'h'))
        return feat - graph.ndata.pop('h') * D_invsqrt

    with g.local_scope():
        D_invsqrt = torch.pow(
            g.in_degrees().float().clamp(min=1), -0.5
        ).unsqueeze(-1).to(x.device)
        h = theta[0] * x
        for k in range(1, len(theta)):
            x = unnLaplacian(x, D_invsqrt, g)
            h += theta[k] * x
    return h


class DynamicWavelets(nn.Module):

    def __init__(self, C: int, num_mix: int, n: int, h_feats: int):
        super().__init__()
        self.C = C
        self.num_mix = num_mix
        self.beta_thetas = beta_wavelet_coefficients(C)
        self.num_beta = len(self.beta_thetas)
        self.theta = nn.Parameter(torch.randn(num_mix, self.num_beta, n, h_feats))

    @property
    def num_wavelets(self) -> int:
        return self.num_beta + self.num_mix

    def mixture_weights(self, h: torch.Tensor):
        if self.theta.shape[0] == 0:
            return None
        X = h.unsqueeze(0).unsqueeze(0)
        X = X.repeat(self.theta.shape[0], self.theta.shape[1], 1, 1)
        return (self.theta * X).sum(dim=(2, 3)).softmax(1)

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor, p_mix) -> list:
        feats = [beta_wavelet_transform(theta, g, h) for theta in self.beta_thetas]
        if p_mix is not None:
            feats += [
                sum(feats[i] * p[i] for i in range(self.num_beta))
                for p in p_mix
            ]
        return feats


class WaveletSpecificFeatureExtraction(nn.Module):

    def __init__(self, num_wavelets: int, h_feats: int):
        super().__init__()
        self.extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_feats, h_feats), nn.ReLU(),
                nn.Linear(h_feats, h_feats),
            )
            for _ in range(num_wavelets)
        ])

    def forward(self, feats: list) -> list:
        return [self.extractors[i](feats[i]) for i in range(len(self.extractors))]


class ChannelConvolutionFusion(nn.Module):

    def __init__(self, num_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        c = num_channels
        self.conv = nn.Sequential(
            nn.Conv1d(c, c, kernel_size, stride, 'same'), nn.BatchNorm1d(c), nn.ReLU(),
            nn.Conv1d(c, c, kernel_size, stride, 'same'), nn.BatchNorm1d(c), nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, feats: list) -> torch.Tensor:
        F_C = torch.cat([x.unsqueeze(1) for x in feats], dim=1)
        return self.conv(F_C)


class DynamicFusion(nn.Module):

    def __init__(self, num_wavelets: int, h_feats: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.wfe = WaveletSpecificFeatureExtraction(num_wavelets, h_feats)
        self.ccf = ChannelConvolutionFusion(num_wavelets, kernel_size, stride)

    def forward(self, feats: list) -> torch.Tensor:
        return self.ccf(self.wfe(feats))


class FDSGAD(BaseModel):

    def __init__(
            self,
            in_nodes: int,
            in_feats: int,
            h_feats: int = 64,
            num_classes: int = 2,
            C: int = 2,
            num_mix: int = 1,
            use_dat: bool = True,
            d_e: int = 16,
            sigma: float = 5.0,
            p_drop: float = 0.1,
            use_bn: bool = True,
    ):
        super().__init__()
        self.use_dat = use_dat

        if use_dat:
            self.dat = DynamicAttributeTransformation(
                d=in_feats, d_e=d_e, sigma=sigma, p_drop=p_drop, use_bn=use_bn,
            )

        self.mlp = nn.Sequential(
            nn.Linear(in_feats, h_feats), nn.ReLU(),
            nn.Linear(h_feats, h_feats), nn.ReLU(),
        )

        num_wavelets = C + 1 + num_mix
        self.df = DynamicFusion(num_wavelets, h_feats)

        self.classifier = nn.Sequential(
            nn.Linear(num_wavelets * h_feats, h_feats), nn.ReLU(),
            nn.Linear(h_feats, num_classes),
        )

        self.dw = DynamicWavelets(C, num_mix, in_nodes, h_feats)

    def forward(self, g: dgl.DGLGraph, in_feat: torch.Tensor) -> torch.Tensor:
        if is_heterograph(g):
            return self._forward_hete(g, in_feat)
        return self._forward_homo(g, in_feat)

    def _forward_homo(self, g: dgl.DGLGraph, in_feat: torch.Tensor) -> torch.Tensor:
        x = self.dat(in_feat) if self.use_dat else in_feat
        h = self.mlp(x)
        p_mix = self.dw.mixture_weights(h)
        feats = self.dw(g, h, p_mix)
        fusion = self.df(feats)
        return self.classifier(fusion)

    def _forward_hete(self, g: dgl.DGLGraph, in_feat: torch.Tensor) -> torch.Tensor:
        x = self.dat(in_feat) if self.use_dat else in_feat
        h = self.mlp(x)
        p_mix = self.dw.mixture_weights(h)
        out_all = []
        for relation in g.canonical_etypes:
            feats = self.dw(g[relation], h, p_mix)
            fusion = self.df(feats)
            out_all.append(self.classifier(fusion))
        return sum(out_all) / len(out_all)

    @classmethod
    def trainfit(cls, args):
        dataset = getattr(datasets, args.dataset)()
        g: dgl.DGLGraph = dataset.g
        print(f'{args.dataset}: {g}')
        g = dataset.split(g, args.ratio)
        print(f'after split: {g}')

        if is_heterograph(g):
            ntype = g.ntypes[0]
            train_mask = g.nodes[ntype].data['train_mask']
            val_mask = g.nodes[ntype].data['val_mask']
            test_mask = g.nodes[ntype].data['test_mask']
            features = g.nodes[ntype].data['feature']
            labels = g.nodes[ntype].data['label']
        else:
            train_mask = g.ndata['train_mask']
            val_mask = g.ndata['val_mask']
            test_mask = g.ndata['test_mask']
            features = g.ndata['feature']
            labels = g.ndata['label']

        print(f'train/val/test: {train_mask.sum().item()} '
              f'{val_mask.sum().item()} {test_mask.sum().item()}')

        g = g.to(args.device)
        features = features.to(args.device)
        labels = labels.to(args.device)
        train_mask = train_mask.to(args.device)
        val_mask = val_mask.to(args.device)
        test_mask = test_mask.to(args.device)

        weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
        print(f'cross entropy weight: {weight:.4f}')

        model = cls(features.shape[0], features.shape[1], **args.model_config).to(args.device)

        params = []
        for name, param in model.named_parameters():
            if name.startswith('dat.') or name == 'dw.theta':
                params.append({'params': param, 'lr': 5e-4})
            else:
                params.append({'params': param, 'lr': 1e-3})
        optimizer = torch.optim.Adam(params)

        print_interval = max(1, args.epoch // 10)
        patience = max(30, args.epoch // 4)
        best_val_auc = 0.0
        best_state = None
        best_epoch = 0
        best_metrics = None
        no_improve = 0

        for e in range(args.epoch):
            model.train()
            out = model(g, features)

            if torch.isnan(out).any():
                print(f'[WARNING] Epoch {e+1}: NaN in output, skip')
                continue

            loss = F.cross_entropy(
                out[train_mask], labels[train_mask],
                weight=torch.tensor([1., weight], device=args.device),
            )
            if torch.isnan(loss):
                print(f'[WARNING] Epoch {e+1}: NaN loss, skip')
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if (e + 1) % print_interval == 0 or e == 0 or e == args.epoch - 1:
                model.eval()
                with torch.no_grad():
                    out = model(g, features)
                if torch.isnan(out).any():
                    print(f'[WARNING] Epoch {e+1}: NaN in eval')
                    continue

                probs = out.softmax(1)[:, 1].cpu()
                probs = torch.nan_to_num(probs, nan=0.0)
                thres = get_thres(labels[val_mask].cpu(), probs[val_mask.cpu()])
                val_auc = skmetrics.roc_auc_score(labels[val_mask].cpu(), probs[val_mask.cpu()])

                probs_test = probs[test_mask.cpu()]
                y_pred = torch.zeros_like(probs_test)
                y_pred[probs_test >= thres] = 1
                y_true = labels[test_mask].cpu()

                auc = skmetrics.roc_auc_score(y_true, probs_test)
                recall = skmetrics.recall_score(y_true, y_pred)
                precision = skmetrics.precision_score(y_true, y_pred, zero_division=0)
                f1_macro = skmetrics.f1_score(y_true, y_pred, average='macro')

                print(f'Epoch {e+1}/{args.epoch}  loss: {loss.item():.5f}  val_auc: {val_auc:.5f}')
                print(f'  auc: {auc:.5f}  recall: {recall:.5f}  '
                      f'precision: {precision:.5f}  f1_macro: {f1_macro:.5f}')

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = e + 1
                    best_state = copy.deepcopy(model.state_dict())
                    best_metrics = {
                        'auc': auc, 'recall': recall,
                        'precision': precision, 'f1_macro': f1_macro,
                    }
                    no_improve = 0
                else:
                    no_improve += print_interval

                if no_improve >= patience:
                    print(f'[Early Stop] no improvement for {no_improve} epochs. '
                          f'best val_auc={best_val_auc:.5f} @ epoch {best_epoch}')
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f'[Best] epoch {best_epoch}  val_auc: {best_val_auc:.5f} -> '
                  f'test_auc: {best_metrics["auc"]:.5f}  '
                  f'recall: {best_metrics["recall"]:.5f}  '
                  f'precision: {best_metrics["precision"]:.5f}  '
                  f'f1_macro: {best_metrics["f1_macro"]:.5f}')
            return model, best_metrics

        final = {'auc': auc, 'recall': recall,
                 'precision': precision, 'f1_macro': f1_macro}
        return model, final


FDSGAD_hete = FDSGAD
