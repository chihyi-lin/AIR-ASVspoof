import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset import ASVspoof2019, InTheWildDataset
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from dataset_asvspoof21 import preprocess_data

"""Adapted from: https://github.com/yzyouzhang/ASVspoof2021_AIR/blob/main/visualize.py"""

def visualize_dev_and_eval(dev_feat, dev_labels, eval_feat, eval_labels,
                           center, seed, out_fold):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    c = ['blue', 'red', 'yellow']

    torch.manual_seed(888)
    if center is None:
        num_centers = 0
    else:
        num_centers, enc_dim = center.shape

    # Using all data points without sampling
    # len(center) = 1, len(dev_feat) = 24844, len(eval_feat) = 31779
    if center is None:
        X = np.concatenate((dev_feat, eval_feat), axis=0)
    else:
        X = np.concatenate((center, dev_feat, eval_feat), axis=0)
    os.environ['PYTHONHASHSEED'] = str(668)
    np.random.seed(668)
    X_tsne = TSNE(random_state=seed, perplexity=40, early_exaggeration=40).fit_transform(X)
    if center is not None:
        center = X_tsne[:num_centers]

    feat_dev = X_tsne[num_centers:num_centers + 24844]
    feat_eval = X_tsne[num_centers + 24844:]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    ex_ratio = pca.explained_variance_ratio_
    center_pca = X_pca[:num_centers]

    feat_pca_dev = X_pca[num_centers:num_centers + 24844]
    feat_pca_eval = X_pca[num_centers + 24844:]

    # t-SNE visualization
    ax1.plot(feat_dev[dev_labels == 0, 0], feat_dev[dev_labels == 0, 1], '.', c=c[0], markersize=1.2)
    ax1.plot(feat_dev[dev_labels == 1, 0], feat_dev[dev_labels == 1, 1], '.', c=c[1], markersize=1.2)
    ax1.axis('off')
    if center is not None:
        ax1.plot(center[:, 0], center[:, 1], 'x', c=c[2], markersize=5)
    ax1.set_title('t_SNE(Dev)')

    plt.setp((ax2), xlim=ax1.get_xlim(), ylim=ax1.get_ylim())
    ax2.plot(feat_eval[eval_labels == 0, 0], feat_eval[eval_labels == 0, 1], '.', c=c[0], markersize=1.2)
    ax2.plot(feat_eval[eval_labels == 1, 0], feat_eval[eval_labels == 1, 1], '.', c=c[1], markersize=1.2)
    ax2.axis('off')
    if center is not None:
        ax2.plot(center[:, 0], center[:, 1], 'x', c=c[2], markersize=5)
    ax2.set_title('t_SNE (Eval)')

    # PCA visualization
    ax3.plot(feat_pca_dev[dev_labels == 0, 0], feat_pca_dev[dev_labels == 0, 1], '.', c=c[0], markersize=1.2)
    ax3.plot(feat_pca_dev[dev_labels == 1, 0], feat_pca_dev[dev_labels == 1, 1], '.', c=c[1], markersize=1.2)
    ax3.axis('off')
    ax3.set_title('PCA (Dev)')

    plt.setp((ax4), xlim=ax3.get_xlim(), ylim=ax3.get_ylim())
    ax4.plot(feat_pca_eval[eval_labels == 0, 0], feat_pca_eval[eval_labels == 0, 1], '.', c=c[0], markersize=1.2)
    ax4.plot(feat_pca_eval[eval_labels == 1, 0], feat_pca_eval[eval_labels == 1, 1], '.', c=c[1], markersize=1.2)
    ax4.axis('off')
    ax4.set_title('PCA (Eval)')

    fig.legend(['real', 'fake'], loc='upper right')
    plt.savefig(os.path.join(out_fold, 'vis_feat.png'), dpi=500, bbox_inches="tight")
    plt.show()
    fig.clf()
    plt.close(fig)


def get_features(feat_model_path, dataset_version="ASVspoof_2019", part=None, frontend=None):
    model = torch.load(feat_model_path)
    if dataset_version == "ASVspoof_2019":
        dataset = ASVspoof2019(access_type="LA", 
                            path_to_features='datasets/ASVspoof2019_LA_Features',
                            path_to_protocol='datasets/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols',
                            part=part,
                            feature="LFCC", 
                            feat_len=750)
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,
                                collate_fn=dataset.collate_fn)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        ip1_loader, idx_loader = [], []
        for i, (feat, audio_fn, tags, labels) in enumerate(tqdm(dataLoader)):
            feat = feat.unsqueeze(1).float().to(device)
            labels = labels.to(device)
            feats, _ = model(feat)
            ip1_loader.append(feats.detach().cpu().numpy())
            idx_loader.append((labels.detach().cpu().numpy()))
        features = np.concatenate(ip1_loader, 0)
        labels = np.concatenate(idx_loader, 0)
    
    elif dataset_version == "ASVspoof_2021":
        trainDataLoader, dataLoader = preprocess_data(
            datasets_paths="datasets/ASVspoof2021/DF",
            train_amount=100000,
            valid_amount=25000,
            batch_size=8,
            frontend=frontend)
        dataLoader.frontend = frontend

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        ip1_loader, idx_loader = [], []
        for i, (feat, labels) in enumerate(tqdm(dataLoader)):
            feat = feat.unsqueeze(1).float().to(device)
            labels = labels.to(device)
            feats, _ = model(feat)
            ip1_loader.append(feats.detach().cpu().numpy())
            idx_loader.append((labels.detach().cpu().numpy()))
        features = np.concatenate(ip1_loader, 0)
        labels = np.concatenate(idx_loader, 0)

    elif dataset_version == "in_the_wild":
        dataset = InTheWildDataset(path_to_features='datasets/in_the_wild_Features',
        path_to_protocol='datasets/meta.csv')
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, 
                                collate_fn=dataset.collate_fn)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        ip1_loader, idx_loader = [], []
        for i, (feat, filename, labels) in enumerate(tqdm(dataLoader)):
            feat = feat.unsqueeze(1).float().to(device)
            labels = labels.to(device)
            feats, _ = model(feat)
            ip1_loader.append(feats.detach().cpu().numpy())
            idx_loader.append((labels.detach().cpu().numpy()))
        features = np.concatenate(ip1_loader, 0)
        labels = np.concatenate(idx_loader, 0)

    return features, labels


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")

    model_dir = "models_asv19_lfcc_100ep/ocsoftmax"     # change model_dir here
    feat_model_path = os.path.join(model_dir, "anti-spoofing_lfcc_model.pt")

    if "ocsoftmax" in model_dir:    
        loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")
        center = torch.load(loss_model_path).center.detach().cpu().numpy()
    else:
        center = None

    dev_feat, dev_labels = get_features(feat_model_path, dataset_version="ASVspoof_2019", part="dev")
    eval_feat, eval_labels = get_features(feat_model_path, dataset_version="in_the_wild", part=None)
    visualize_dev_and_eval(dev_feat, dev_labels, eval_feat, eval_labels, center, 88, model_dir)
