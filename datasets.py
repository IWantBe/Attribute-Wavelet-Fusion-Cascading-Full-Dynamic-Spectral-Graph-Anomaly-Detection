import torch
import dgl
from dgl.data import FraudAmazonDataset, FraudYelpDataset
import scipy.io as sio
import scipy.sparse as sp
import sklearn.model_selection as skselection
from pathlib import Path

datasets_path = Path(__file__).resolve().parent


class BaseDataset:

    def __init__(self, *args, **kwargs):
        pass

    def split(self, g: dgl.DGLGraph, ratio: list) -> dgl.DGLGraph:
        assert len(ratio) == 3
        assert sum(ratio) == 1
        for i in ratio:
            assert 0 <= i <= 1

        num_nodes = g.num_nodes()
        labels = g.ndata['label']

        s = ['train_mask', 'val_mask', 'test_mask']
        if 1 in ratio:
            idx = ratio.index(1)
            for i in range(3):
                if i == idx:
                    g.ndata[s[i]] = torch.ones(num_nodes).bool()
                else:
                    g.ndata[s[i]] = torch.zeros(num_nodes).bool()

        elif 0 in ratio:
            idx = ratio.index(0)
            g.ndata[s[idx]] = torch.zeros(num_nodes).bool()

            idx1, idx2 = 0, 0
            for i in range(3):
                idx1 = i
                if idx1 != idx:
                    break
            for i in range(3):
                idx2 = i
                if idx2 != idx and idx2 != idx1:
                    break

            indices = list(range(num_nodes))
            x_train, x_test, _, _ = skselection.train_test_split(
                indices,
                labels,
                stratify=labels,
                train_size=ratio[idx1],
                random_state=2,
                shuffle=True,
            )
            mask1 = torch.zeros(num_nodes).bool()
            mask2 = torch.zeros(num_nodes).bool()
            mask1[x_train] = True
            mask2[x_test] = True

            g.ndata[s[idx1]] = mask1
            g.ndata[s[idx2]] = mask2

        else:
            index = list(range(num_nodes))
            idx_train, idx_rest, _, y_rest = skselection.train_test_split(
                index,
                labels,
                stratify=labels,
                train_size=ratio[0],
                random_state=2,
                shuffle=True,
            )
            idx_valid, idx_test, _, _ = skselection.train_test_split(
                idx_rest,
                y_rest,
                stratify=y_rest,
                train_size=ratio[1] / (ratio[1] + ratio[2]),
                random_state=2,
                shuffle=True,
            )
            train_mask = torch.zeros(num_nodes).bool()
            val_mask = torch.zeros(num_nodes).bool()
            test_mask = torch.zeros(num_nodes).bool()

            train_mask[idx_train] = True
            val_mask[idx_valid] = True
            test_mask[idx_test] = True

            g.ndata[s[0]] = train_mask
            g.ndata[s[1]] = val_mask
            g.ndata[s[2]] = test_mask

        return g


def _load_dgl_graph(name: str) -> dgl.DGLGraph:
    path = datasets_path / name
    if not path.exists():
        raise FileNotFoundError(f'{name} dataset not found at {path}')
    g = dgl.load_graphs(str(path))[0][0]
    g.ndata['label'] = g.ndata['label'].long().squeeze(-1)
    g.ndata['feature'] = g.ndata['feature'].float()
    return g


def _load_pyg_graph(name: str) -> dgl.DGLGraph:
    pt_path = datasets_path / name / f'{name}.pt'
    if not pt_path.exists():
        raise FileNotFoundError(f'{name} dataset not found at {pt_path}')

    data = torch.load(str(pt_path), map_location='cpu')
    try:
        x = data.x
        y = data.y
        edge_index = data.edge_index
    except AttributeError:
        store = data['_store'] if '_store' in data else data
        x = store['x']
        y = store['y']
        edge_index = store['edge_index']

    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long).view(-1)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    assert edge_index.dim() == 2 and edge_index.size(0) == 2

    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.size(0))
    g = dgl.add_self_loop(g)
    g.ndata['feature'] = x
    g.ndata['label'] = y
    return g


class tfinance(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g = dgl.load_graphs(str(datasets_path / 'tfinance'))[0][0]
        self.g.ndata['label'] = self.g.ndata['label'].argmax(1)
        self.g.ndata['label'] = self.g.ndata['label'].long().squeeze(-1)
        self.g.ndata['feature'] = self.g.ndata['feature'].float()


class tolokers(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g = _load_dgl_graph('tolokers')


class questions(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g = _load_dgl_graph('questions')


class weibo(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g = _load_pyg_graph('weibo')


class reddit(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g = _load_pyg_graph('reddit')


class photo(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mat_path = datasets_path / 'photo.mat'
        if not mat_path.exists():
            raise FileNotFoundError(f'photo dataset not found at {mat_path}')

        data = sio.loadmat(str(mat_path))

        A = data['Network']
        if sp.issparse(A):
            A = A.tocoo()
        else:
            A = sp.coo_matrix(A)
        src, dst = torch.from_numpy(A.row).long(), torch.from_numpy(A.col).long()
        num_nodes = A.shape[0]

        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g = dgl.add_self_loop(g)

        X = data['Attributes']
        if sp.issparse(X):
            X = X.toarray()
        g.ndata['feature'] = torch.from_numpy(X).float()
        g.ndata['label'] = torch.from_numpy(data['Label'].flatten()).long()

        self.g = g


class elliptic(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ep = datasets_path / 'elliptic'
        if not ep.exists():
            raise FileNotFoundError(f'elliptic dataset not found at {ep}')

        with open(str(ep / 'elliptic_txs_classes.csv')) as f:
            labels = f.readlines()
            labels.pop(0)
            labels = list(map(lambda x: x.strip().split(','), labels))

            for i, (node, label) in enumerate(labels):
                if label == 'unknown':
                    labels[i][1] = 2
                elif label == '2':
                    labels[i][1] = 1
                elif label == '1':
                    labels[i][1] = 0
                labels[i] = list(map(lambda x: int(x), labels[i]))

            labels.sort(key=lambda x: x[0])

            di = {}
            for i in range(len(labels)):
                di[labels[i][0]] = i

            labels = torch.tensor(labels, dtype=torch.long)
            marks = labels[:, 1] != 2
            labels[labels[:, 1] == 2, 1] = 0
            labels = labels[:, 1]

        with open(str(ep / 'elliptic_txs_features.csv')) as f:
            features = f.readlines()
            features = list(map(lambda x: x.strip().split(','), features))
            features = [list(map(lambda x: float(x), feature)) for feature in features]
            features.sort(key=lambda x: x[0])

            time_steps = torch.tensor([feature[1] for feature in features], dtype=torch.long)

            features = [feature[2:] for feature in features]
            features = torch.tensor(features, dtype=torch.float)

            self.train_mask = (time_steps <= 25) & marks
            self.val_mask = (time_steps > 25) & (time_steps <= 34) & marks
            self.test_mask = (time_steps > 34) & marks

        with open(str(ep / 'elliptic_txs_edgelist.csv')) as f:
            edges = f.readlines()
            edges.pop(0)
            edges = list(map(lambda x: x.strip().split(','), edges))
            a = [di[int(edge[0])] for edge in edges]
            b = [di[int(edge[1])] for edge in edges]

        self.g = dgl.graph((a, b))
        self.g.ndata['feature'] = features
        self.g.ndata['label'] = labels
        self.g = dgl.add_self_loop(self.g)

        self.g.ndata['label'] = self.g.ndata['label'].long().squeeze(-1)
        self.g.ndata['feature'] = self.g.ndata['feature'].float()

    def split(self, g: dgl.DGLGraph, ratio: list) -> dgl.DGLGraph:
        g.ndata['train_mask'] = self.train_mask.clone().detach()
        g.ndata['val_mask'] = self.val_mask.clone().detach()
        g.ndata['test_mask'] = self.test_mask.clone().detach()
        return g


class amazon(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.g = FraudAmazonDataset(raw_dir=str(datasets_path))[0]
        self.g = dgl.to_homogeneous(self.g, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        self.g = dgl.add_self_loop(self.g)

        self.g.ndata['label'] = self.g.ndata['label'].long().squeeze(-1)
        self.g.ndata['feature'] = self.g.ndata['feature'].float()


class amazon_hete(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.g = FraudAmazonDataset(raw_dir=str(datasets_path))[0]

        self.g.ndata['label'] = self.g.ndata['label'].long().squeeze(-1)
        self.g.ndata['feature'] = self.g.ndata['feature'].float()


class yelp(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.g = FraudYelpDataset(raw_dir=str(datasets_path))[0]
        self.g = dgl.to_homogeneous(self.g, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        self.g = dgl.add_self_loop(self.g)

        self.g.ndata['label'] = self.g.ndata['label'].long().squeeze(-1)
        self.g.ndata['feature'] = self.g.ndata['feature'].float()


class yelp_hete(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.g = FraudYelpDataset(raw_dir=str(datasets_path))[0]

        self.g.ndata['label'] = self.g.ndata['label'].long().squeeze(-1)
        self.g.ndata['feature'] = self.g.ndata['feature'].float()
