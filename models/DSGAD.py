import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import sklearn.metrics as skmetrics

import datasets
from .base import BaseModel, get_thres
from .FDSGAD import beta_wavelet_coefficients, beta_wavelet_transform


class DSGAD(BaseModel):

    def __init__(
            self,
            in_nodes: int,
            in_feats: int,
            h_feats: int = 64,
            num_classes: int = 2,
            C: int = 2,
            num_mix: int = 1,
    ):
        super().__init__()

        self.thetas = beta_wavelet_coefficients(C)
        self.num_filters = len(self.thetas)

        self.input = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
        )

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats),
        ) for _ in range(self.num_filters + num_mix)])

        c = self.num_filters + num_mix
        ks = 3
        stride = 1
        self.conv = nn.Sequential(
            nn.Conv1d(c, c, ks, stride, 'same'),
            nn.BatchNorm1d(c),
            nn.ReLU(),
            nn.Conv1d(c, c, ks, stride, 'same'),
            nn.BatchNorm1d(c),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(c * h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, num_classes),
        )

        self.weights = nn.Parameter(torch.randn(num_mix, self.num_filters, in_nodes, h_feats))

    def forward(self, g: dgl.DGLGraph, in_feat: torch.Tensor):
        h = self.input(in_feat)

        if self.weights.shape[0] > 0:
            X = h.unsqueeze(0).unsqueeze(0)
            X = X.repeat(self.weights.shape[0], self.weights.shape[1], 1, 1)
            weights = self.weights * X
            weights = weights.sum(dim=(2, 3))

        h = [beta_wavelet_transform(theta, g, h) for theta in self.thetas]
        if self.weights.shape[0] > 0:
            mix = [sum([h[i] * w[i] for i in range(self.num_filters)]) for w in weights.softmax(1)]
            h += mix

        h = [self.fcs[i](h[i]) for i in range(len(self.fcs))]

        h = [x.unsqueeze(1) for x in h]
        h = torch.cat(h, 1)
        h = self.conv(h)
        h = self.classifier(h)

        return h

    @classmethod
    def trainfit(cls, args):
        dataset = getattr(datasets, args.dataset)()
        g: dgl.DGLGraph = dataset.g
        print(f'{args.dataset}: {g}')
        g = dataset.split(g, args.ratio)
        print(f'after split: {g}')

        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        print(f'train/val/test samples: {train_mask.sum().item()} {val_mask.sum().item()} {test_mask.sum().item()}')

        g = g.to(args.device)
        features = g.ndata['feature']
        labels = g.ndata['label']

        weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
        print(f'cross entropy weight: {weight}')

        model = cls(features.shape[0], features.shape[1], **args.model_config).to(args.device)

        params = []
        for name, param in model.named_parameters():
            if name == 'weights':
                params.append({'params': param, 'lr': 1e-1})
            else:
                params.append({'params': param, 'lr': 1e-3})
        optimizer = torch.optim.Adam(params)

        for e in range(args.epoch):

            model.train()
            out = model(g, features)
            loss = F.cross_entropy(
                out[train_mask],
                labels[train_mask],
                weight=torch.tensor([1., weight], device=args.device),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out = model(g, features)
            probs = out.softmax(1)[:, 1].cpu()
            thres = get_thres(labels[val_mask].cpu(), probs[val_mask])

            probs = probs[test_mask]
            y_pred = torch.zeros_like(probs)
            y_pred[probs >= thres] = 1
            y_true = labels[test_mask].cpu()

            auc = skmetrics.roc_auc_score(y_true, probs)
            recall = skmetrics.recall_score(y_true, y_pred)
            precision = skmetrics.precision_score(y_true, y_pred)
            f1_macro = skmetrics.f1_score(y_true, y_pred, average='macro')

            print(f'Epoch {e+1}, loss: {loss.item():.5f}')
            print('auc  recall  precision  f1_macro')
            print(f'{auc:.5f}  {recall:.5f}  {precision:.5f}  {f1_macro:.5f}')

        return model, {
            'auc': auc,
            'recall': recall,
            'precision': precision,
            'f1_macro': f1_macro,
        }


class DSGAD_hete(DSGAD):

    def forward(self, g: dgl.DGLGraph, in_feat: torch.Tensor):
        h = self.input(in_feat)

        if self.weights.shape[0] > 0:
            X = h.unsqueeze(0).unsqueeze(0)
            X = X.repeat(self.weights.shape[0], self.weights.shape[1], 1, 1)
            weights = self.weights * X
            weights = weights.sum(dim=(2, 3))

        h_all = []
        for relation in g.canonical_etypes:

            hh = [beta_wavelet_transform(theta, g[relation], h) for theta in self.thetas]
            if self.weights.shape[0] > 0:
                mix = [sum([hh[i] * w[i] for i in range(self.num_filters)]) for w in weights.softmax(1)]
                hh += mix

            hh = [self.fcs[i](hh[i]) for i in range(len(self.fcs))]

            hh = [x.unsqueeze(1) for x in hh]
            hh = torch.cat(hh, 1)
            hh = self.conv(hh)
            hh = self.classifier(hh)

            h_all.append(hh)

        h = sum(h_all) / len(h_all)

        return h
