import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from dataset import ASVspoof2019, InTheWildDataset
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import eval_metrics as em
import numpy as np
from evaluate_in_the_wild import compute_eer_in_the_wild

"""plot_t_SNE function is adapted from: https://gist.github.com/XIAOYixuan/81f6522c84f62ab644a05b4e3d21ffb7"""

def test_model(feat_model_path, loss_model_path, test_set, part, add_loss, device):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    model = torch.load(feat_model_path, map_location="cuda")
    model = model.to(device)
    loss_model = torch.load(loss_model_path) if add_loss != "softmax" else None

    if test_set == 'ASVspoof2019':
        test_obj = ASVspoof2019("LA", "datasets/ASVspoof2019_LA_Features",
                                "datasets/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols", part,
                                "LFCC", feat_len=750, padding="repeat")    
    elif test_set == 'in_the_wild':
        test_obj = InTheWildDataset(path_to_features='datasets/in_the_wild_Features',
        path_to_protocol='datasets/meta.csv')
    
    testDataLoader = DataLoader(test_obj, batch_size=32, shuffle=False, num_workers=0,
                                    collate_fn=test_obj.collate_fn)

    model.eval()

    if test_set == 'ASVspoof2019':
        if not os.path.exists(os.path.join(dir_path, 'checkpoint_cm_score.txt')):
            print('create checkpoint_cm_score.txt and run evaluation:')
            with open(os.path.join(dir_path, 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
                for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
                    lfcc = lfcc.unsqueeze(1).float().to(device)
                    tags = tags.to(device)
                    labels = labels.to(device)

                    feats, lfcc_outputs = model(lfcc)

                    score = F.softmax(lfcc_outputs)[:, 0]

                    if add_loss == "ocsoftmax":
                        ang_isoloss, score = loss_model(feats, labels)
                    elif add_loss == "amsoftmax":
                        outputs, moutputs = loss_model(feats, labels)
                        score = F.softmax(outputs, dim=1)[:, 0]

                    for j in range(labels.size(0)):
                        cm_score_file.write(
                            '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                                "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                                score[j].item()))
        
        eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, 'checkpoint_cm_score.txt'),
                                                    "datasets/ASVspoof2019_LA")
        return eer_cm, min_tDCF
    
    elif test_set == 'in_the_wild':
        with open(os.path.join(dir_path, 'in_the_wild_score.txt'), 'w') as in_the_wild_score_file:
            all_feats = torch.Tensor([]).to(device)
            all_labels = torch.Tensor([]).to(device)  

            for i, (lfcc, filename, labels) in enumerate(tqdm(testDataLoader)):
  
                with torch.no_grad():
                    lfcc = lfcc.unsqueeze(1).float().to(device)
                    labels = labels.to(device)

                    feats, lfcc_outputs = model(lfcc)
                    score = F.softmax(lfcc_outputs, dim=1)[:, 0]

                    if add_loss == "ocsoftmax":
                        ang_isoloss, score = loss_model(feats, labels)
                    elif add_loss == "amsoftmax":
                        outputs, moutputs = loss_model(feats, labels)
                        score = F.softmax(outputs, dim=1)[:, 0]
                    
                    # for plotting t-SNE
                    feats = F.normalize(feats, p=2, dim=1)
                    all_feats = torch.concat([all_feats, feats], dim=0)
                    all_labels = torch.concat([all_labels, labels], dim=0)
                    
                    for j in range(labels.size(0)):
                        in_the_wild_score_file.write(
                            '%s %s %s\n' % (filename[j], 
                                            "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                            score[j].item()))
            
        thresh, eer, fpr, tpr = compute_eer_in_the_wild(os.path.join(dir_path, 'in_the_wild_score.txt'))
        print(f'EER In-the-wild: {eer}, thresh: {-thresh}')
        # eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, 'in_the_wild_score.txt'),
        #                                     "datasets/ASVspoof2019_LA")

        return all_feats, all_labels                      


def test(model_dir, add_loss, device, test_set):
    model_path = os.path.join(model_dir, "anti-spoofing_lfcc_model.pt")
    loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")
    all_feats, all_labels = test_model(model_path, loss_model_path, test_set, "eval", add_loss, device)
    return all_feats, all_labels


def project_to_2d(all_feats, all_labels):
    print("all_feats:", all_feats.shape) # tensor [n, dim]
    print("all_labels:", all_labels.shape) # tensor [n]
    all_feats = all_feats.cpu().numpy()
    all_labels = all_labels.cpu().numpy()
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0)
    all_feats = tsne.fit_transform(all_feats)

    pos_feats = all_feats[all_labels == 0] # it is detected as a real audio
    neg_feats = all_feats[all_labels == 1]
    return pos_feats, neg_feats


def plot_t_SNE(real_dots, fake_dots, model_dir):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(real_dots[:, 0], real_dots[:, 1], c="blue", label="real", s=1)
    plt.scatter(fake_dots[:, 0], fake_dots[:, 1], c="red", label="fake", s=1)
    plt.legend()
    # save
    plt.savefig(os.path.join(model_dir, "t-SNE_plot.png"))


def test_individual_attacks(cm_score_file):
    asv_score_file = os.path.join('/data/neil/DS_10283_3336',
                                  'LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt')

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    other_cm_scores = -cm_scores

    eer_cm_lst, min_tDCF_lst = [], []
    for attack_idx in range(7,20):
        # Extract target, nontarget, and spoof scores from the ASV scores
        tar_asv = asv_scores[asv_keys == 'target']
        non_asv = asv_scores[asv_keys == 'nontarget']
        spoof_asv = asv_scores[asv_sources == 'A%02d' % attack_idx]

        # Extract bona fide (real human) and spoof scores from the CM scores
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_sources == 'A%02d' % attack_idx]

        # EERs of the standalone systems and fix ASV operating point to EER threshold
        eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

        other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_sources == 'A%02d' % attack_idx])[0]

        [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

        if eer_cm < other_eer_cm:
            # Compute t-DCF
            tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model,
                                                        True)
            # Minimum t-DCF
            min_tDCF_index = np.argmin(tDCF_curve)
            min_tDCF = tDCF_curve[min_tDCF_index]

        else:
            tDCF_curve, CM_thresholds = em.compute_tDCF(other_cm_scores[cm_keys == 'bonafide'],
                                                        other_cm_scores[cm_sources == 'A%02d' % attack_idx],
                                                        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)
            # Minimum t-DCF
            min_tDCF_index = np.argmin(tDCF_curve)
            min_tDCF = tDCF_curve[min_tDCF_index]
        eer_cm_lst.append(min(eer_cm, other_eer_cm))
        min_tDCF_lst.append(min_tDCF)

    return eer_cm_lst, min_tDCF_lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_dir', type=str, help="path to the trained model", default="./models/ocsoftmax")
    parser.add_argument('-l', '--loss', type=str, default="ocsoftmax",
                        choices=["softmax", 'amsoftmax', 'ocsoftmax'], help="loss function")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument('-t', '--test_set', type=str, help="choose a test dataset", default='in_the_wild')
    parser.add_argument('-s', '--score_file', type=str, help="score file of the evaluation set", default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_feats, all_labels = test(args.model_dir, args.loss, args.device, args.test_set)
    # pos_feats, neg_feats = project_to_2d(all_feats, all_labels)
    # plot_t_SNE(pos_feats, neg_feats, args.model_dir)
    
    # compute_eer_from_score_doc
    # thresh, eer, fpr, tpr = compute_eer_in_the_wild(args.score_file)
    # print(f'EER In-the-wild: {eer:.4f}, thresh; {-thresh}')

    # eer_cm_lst, min_tDCF_lst = test_individual_attacks(os.path.join(args.model_dir, 'checkpoint_cm_score.txt'))
    # print(eer_cm_lst)
    # print(min_tDCF_lst)
