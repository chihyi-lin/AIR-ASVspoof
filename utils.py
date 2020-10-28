import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import ASVspoof2019
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import eval_metrics as em

def test_checkpoint_model(feat_model_path, loss_model_path, part, add_loss, vis=False):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
        epoch_num = int(basename.split("_")[-1])
    else:
        dir_path = dirname(feat_model_path)
        epoch_num = 0
    model = torch.load(feat_model_path)
    loss_model = torch.load(loss_model_path) if add_loss is not None else None
    test_set = ASVspoof2019("LA", "/data/neil/DS_10283_3336/", "/dataNVME/neil/ASVspoof2019LAFeatures/",
                            "/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part,
                            "LFCC", feat_len=750, pad_chop=False, padding="repeat")
    testDataLoader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0,
                                collate_fn=test_set.collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with open(os.path.join(dir_path, 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
        ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
        for i, (cqcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
            cqcc = cqcc.unsqueeze(1).float().to(device)
            tags = tags.to(device)
            labels = labels.to(device)

            feats, cqcc_outputs = model(cqcc)

            score = cqcc_outputs[:, 0]

            ip1_loader.append(feats.detach().cpu())
            idx_loader.append((labels.detach().cpu()))
            tag_loader.append((tags.detach().cpu()))

            if add_loss in ["isolate", "iso_sq"]:
                score = torch.norm(feats - loss_model.center, p=2, dim=1)
            elif add_loss == "ang_iso":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "lgcl":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

    eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, 'checkpoint_cm_score.txt'),
                                            "/data/neil/DS_10283_3336/")

    if vis:
        feat = torch.cat(ip1_loader, 0)
        labels = torch.cat(idx_loader, 0)
        tags = torch.cat(tag_loader, 0)
        if add_loss == "isolate":
            centers = loss_model.center
        elif add_loss == "multi_isolate":
            centers = loss_model.centers
        elif add_loss == "ang_iso":
            centers = loss_model.center
        else:
            centers = torch.mean(feat[labels == 0], dim=0, keepdim=True)
        torch.save(feat, os.path.join(dir_path, 'feat_%d.pt' % epoch_num))
        torch.save(tags, os.path.join(dir_path, 'tags_%d.pt' % epoch_num))
        visualize(feat.data.cpu().numpy(), tags.data.cpu().numpy(), labels.data.cpu().numpy(),
                  centers.data.cpu().numpy(), epoch_num, part, dir_path)

    return eer_cm, min_tDCF

def visualize_dev_and_eval(dev_feat, dev_labels, eval_feat, eval_labels, center, epoch, out_fold):
    # visualize which experiemnt which epoch
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    # plt.ion()
    # c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
    #      '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    c = ['#ff0000', '#003366', '#ffff00']
    c_tag = ['#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900',
             '#009900', '#009999', '#00ff00', '#990000', '#999900', '#ff0000',
             '#003366', '#ffff00', '#f0000f', '#0f00f0', '#00ffff', '#0000ff', '#ff00ff']
    # plt.clf()
    torch.manual_seed(668)
    num_centers, enc_dim = center.shape
    ind_dev = torch.randperm(dev_feat.shape[0])[:5000].numpy()
    ind_eval = torch.randperm(eval_feat.shape[0])[:5000].numpy()

    dev_feat_sample = dev_feat[ind_dev]
    eval_feat_sample = eval_feat[ind_eval]
    dev_lab_sam = dev_labels[ind_dev]
    eval_lab_sam = eval_labels[ind_eval]
    if enc_dim > 2:
        X = np.concatenate((center, dev_feat_sample, eval_feat_sample), axis=0)
        os.environ['PYTHONHASHSEED'] = str(668)
        np.random.seed(668)
        X_tsne = TSNE(random_state=999, perplexity=40, early_exaggeration=40).fit_transform(X)
        center = X_tsne[:num_centers]
        feat_dev = X_tsne[num_centers:num_centers+5000]
        feat_eval = X_tsne[num_centers + 5000:]
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        ex_ratio = pca.explained_variance_ratio_
        center_pca = X_pca[:num_centers]
        feat_pca_dev = X_pca[num_centers:num_centers+5000]
        feat_pca_eval = X_pca[num_centers + 5000:]
    else:
        center_pca = center
        feat_dev = dev_feat_sample
        feat_eval = eval_feat_sample
        feat_pca_dev = feat_dev
        feat_pca_eval = feat_eval
        ex_ratio = [0.5, 0.5]
    # t-SNE visualization
    ax1.plot(feat_dev[dev_lab_sam == 0, 0], feat_dev[dev_lab_sam == 0, 1], '.', c=c[0], markersize=1)
    ax1.plot(feat_dev[dev_lab_sam == 1, 0], feat_dev[dev_lab_sam == 1, 1], '.', c=c[1], markersize=1)
    ax1.axis('off')
    # ax1.plot(center[:, 0], center[:, 1], 'x', c=c[2], markersize=5)
    # ax2.plot(center[:, 0], center[:, 1], 'x', c=c[2], markersize=5)
    plt.setp((ax2), xlim=ax1.get_xlim(), ylim=ax1.get_ylim())
    ax2.plot(feat_eval[eval_lab_sam == 0, 0], feat_eval[eval_lab_sam == 0, 1], '.', c=c[0], markersize=1)
    ax2.plot(feat_eval[eval_lab_sam == 1, 0], feat_eval[eval_lab_sam == 1, 1], '.', c=c[1], markersize=1)
    ax2.axis('off')
    # ax1.legend(['genuine', 'spoofing', 'center'])
    # PCA visualization
    ax3.plot(feat_pca_dev[dev_lab_sam == 0, 0], feat_pca_dev[dev_lab_sam == 0, 1], '.', c=c[0], markersize=1)
    ax3.plot(feat_pca_dev[dev_lab_sam == 1, 0], feat_pca_dev[dev_lab_sam == 1, 1], '.', c=c[1], markersize=1)
    # ax3.plot(center_pca[:, 0], center_pca[:, 1], 'x', c=c[2], markersize=5)
    ax3.axis('off')
    plt.setp((ax4), xlim=ax3.get_xlim(), ylim=ax3.get_ylim())
    ax4.plot(feat_pca_eval[eval_lab_sam == 0, 0], feat_pca_eval[eval_lab_sam == 0, 1], '.', c=c[0], markersize=1)
    ax4.plot(feat_pca_eval[eval_lab_sam == 1, 0], feat_pca_eval[eval_lab_sam == 1, 1], '.', c=c[1], markersize=1)
    ax4.axis('off')
    # ax4.legend(['genuine', 'spoofing', 'center'])
    fig.suptitle("Generalization Visualization of Epoch %d, %.5f, %.5f" % (epoch, ex_ratio[0], ex_ratio[1]))
    plt.savefig(os.path.join(out_fold, '_vis_feat_epoch=%d.jpg' % epoch))
    plt.show()
    fig.clf()
    plt.close(fig)

def get_features(feat_model_path, part):
    model = torch.load(feat_model_path)
    dataset = ASVspoof2019("LA", "/data/neil/DS_10283_3336/", "/dataNVME/neil/ASVspoof2019LAFeatures/",
                            "/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part,
                            "LFCC", feat_len=750, pad_chop=False, padding="repeat")
    dataLoader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                                collate_fn=dataset.collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
    for i, (cqcc, audio_fn, tags, labels) in enumerate(tqdm(dataLoader)):
        cqcc = cqcc.unsqueeze(1).float().to(device)
        tags = tags.to(device)
        labels = labels.to(device)
        feats, _ = model(cqcc)
        ip1_loader.append(feats.detach().cpu().numpy())
        idx_loader.append((labels.detach().cpu().numpy()))
        tag_loader.append((tags.detach().cpu().numpy()))
    features = np.concatenate(ip1_loader, 0)
    labels = np.concatenate(idx_loader, 0)
    gen_feats = features[labels==0]
    return features, labels, gen_feats

def predict_with_OCSVM(feat_model_path):
    from sklearn.svm import OneClassSVM
    _, _, train_gen_feats = get_features(feat_model_path, "train")
    clf = OneClassSVM(gamma='auto', degree=256, nu=0.05).fit(train_gen_feats)
    dev_features, dev_labels, _ = get_features(feat_model_path, "dev")
    eval_features, eval_labels, _ = get_features(feat_model_path, "eval")
    dev_scores = clf.score_samples(dev_features)
    eval_scores = clf.score_samples(eval_features)
    dev_eer = em.compute_eer(dev_scores[dev_labels==0], dev_scores[dev_labels==1])
    eval_eer = em.compute_eer(eval_scores[eval_labels==0], eval_scores[eval_labels==1])

    clf2 = OneClassSVM(gamma='auto', degree=256, nu=0.5).fit(train_gen_feats)
    dev_scores = clf2.score_samples(dev_features)
    eval_scores = clf2.score_samples(eval_features)
    dev_eer2 = em.compute_eer(dev_scores[dev_labels == 0], dev_scores[dev_labels == 1])
    eval_eer2 = em.compute_eer(eval_scores[eval_labels == 0], eval_scores[eval_labels == 1])

    clf3 = OneClassSVM(gamma='auto', degree=256, nu=0.95).fit(train_gen_feats)
    dev_scores = clf3.score_samples(dev_features)
    eval_scores = clf3.score_samples(eval_features)
    dev_eer3 = em.compute_eer(dev_scores[dev_labels == 0], dev_scores[dev_labels == 1])
    eval_eer3 = em.compute_eer(eval_scores[eval_labels == 0], eval_scores[eval_labels == 1])

    return dev_eer[0], eval_eer[0], dev_eer2[0], eval_eer2[0], dev_eer3[0], eval_eer3[0]

if __name__ == "__main__":
    # args, (a, b, c, d, e, f) = read_args_json("models/cqt_cnn1")
    #
    # split_dict = {'125_346': ["A01", "A02", "A05"], '135_246': ["A01", "A03", "A05"], '145_236': ["A01", "A04", "A05"],
    #               '235_146': ["A02", "A03", "A05"], '245_136': ["A02", "A04", "A05"], '345_126': ["A03", "A04", "A05"],
    #               '126_345': ["A01", "A02", "A06"], '136_245': ["A01", "A03", "A06"], '146_235': ["A01", "A04", "A06"],
    #               '236_145': ["A02", "A03", "A06"], '246_135': ["A02", "A04", "A06"], '346_125': ["A03", "A04", "A06"]}
    # df_train = pd.read_csv("/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm." + "train" + ".trl.txt",
    #                  names=["speaker", "filename", "-", "attack", "label"], sep=" ")
    # df_dev = pd.read_csv("/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm." + "dev" + ".trl.txt",
    #                  names=["speaker", "filename", "-", "attack", "label"], sep=" ")
    # create_new_split(df_train, df_dev, split_dict)

    # feat_model_path = "/home/neil/AIR-ASVspoof/models0922/ang_iso/checkpoint/anti-spoofing_cqcc_model_19.pt"
    # loss_model_path = "/home/neil/AIR-ASVspoof/models0922/ang_iso/checkpoint/anti-spoofing_loss_model_19.pt"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # test_checkpoint_model(feat_model_path, loss_model_path, "eval", "ang_iso")

    # b = []
    # for i in range(50, 120, 2):
    #     feat_model_path = "/data/neil/antiRes/models1007/iso_loss4/checkpoint/anti-spoofing_cqcc_model_%d.pt" % i
    #     loss_model_path = "/data/neil/antiRes/models1007/iso_loss4/checkpoint/anti-spoofing_loss_model_%d.pt" % i
    #     eer = test_fluctuate_eer(feat_model_path, loss_model_path, "eval", "isolate")
    #     print(eer)
    #     b.append(eer)

    # feat_model_path = "/data/neil/antiRes/models1007/ce/anti-spoofing_cqcc_model.pt"
    # loss_model_path = "/data/neil/antiRes/models1007/ce/anti-spoofing_loss_model.pt"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # test_checkpoint_model(feat_model_path, loss_model_path, "eval", None)
    # print(eer)

    feat_model_path = "/data/neil/antiRes/models1020/softmax/checkpoint/anti-spoofing_cqcc_model_%d.pt" % 78
    loss_model_path = "/data/neil/antiRes/models1020/softmax/checkpoint/anti-spoofing_loss_model_%d.pt" % 78
    dev_eer, eval_eer, dev_eer2, eval_eer2, dev_eer3, eval_eer3 = predict_with_OCSVM(feat_model_path)
    print(dev_eer, eval_eer, dev_eer2, eval_eer2, dev_eer3, eval_eer3)
