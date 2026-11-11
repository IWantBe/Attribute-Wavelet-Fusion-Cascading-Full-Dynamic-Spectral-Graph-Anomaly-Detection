MODEL_DEFAULT_CONFIGS = {
    'FDSGAD': {
        'h_feats': 64,
        'C': 2,
        'num_mix': 1,
        'use_dat': True,
        'd_e': 16,
        'sigma': 5.0,
        'p_drop': 0.1,
        'use_bn': True,
    },
    'DSGAD': {
        'h_feats': 64,
        'C': 2,
        'num_mix': 1,
    },
    'GAAP': {
        'use_dyple': True,
        'n_bins': 16,
        'd_feat_emb': 8,
        'dyple_dropout': 0.1,
        'dyple_use_bn': True,
        'gnn_layers': 2,
        'gnn_dropout': 0.1,
        'gnn_use_bn': True,
        'gnn_use_res': True,
    },
    'SECGFD': {
        'hid_dim': 64,
        'd': 2,
        'high_order': 2,
        'lr': 0.01,
        'weight_decay': 1e-4,
        'lambda_lec': 0.3,
        'lec_margin': 0.5,
        'dropout': 0.0,
    },
}

DATASET_OVERRIDES = {
    'elliptic': {
        'FDSGAD': {'d_e': 4, 'sigma': 1.0},
        'GAAP': {'n_bins': 8, 'd_feat_emb': 4},
    },
}


def get_model_config(model_name: str, dataset_name: str, extra: dict = None) -> dict:
    base_name = model_name[:-5] if model_name.endswith('_hete') else model_name
    config = dict(MODEL_DEFAULT_CONFIGS.get(base_name, {}))
    config.update(DATASET_OVERRIDES.get(dataset_name, {}).get(base_name, {}))
    if extra:
        config.update(extra)
    return config
