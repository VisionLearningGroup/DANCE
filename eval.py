import torch
import torch.nn.functional as F
import numpy as np
import logging

def test(step, dataset_test, filename, n_share, unk_class, G, C1, threshold):
    G.eval()
    C1.eval()
    correct = 0
    correct_close = 0
    size = 0
    class_list = [i for i in range(n_share)]
    class_list.append(unk_class)
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    per_class_correct_cls = np.zeros((n_share + 1)).astype(np.float32)
    all_pred = []
    all_gt = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t, path_t = data[0], data[1], data[2]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            feat = G(img_t)
            out_t = C1(feat)
            out_t = F.softmax(out_t)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            pred = out_t.data.max(1)[1]
            k = label_t.data.size()[0]
            pred_cls = pred.cpu().numpy()
            pred = pred.cpu().numpy()

            pred_unk = np.where(entr > threshold)
            pred[pred_unk[0]] = n_share
            all_gt += list(label_t.data.cpu().numpy())
            all_pred += list(pred)
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == i)
                correct_ind_close = np.where(pred_cls[t_ind[0]] == i)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_correct_cls[i] += float(len(correct_ind_close[0]))
                per_class_num[i] += float(len(t_ind[0]))
                correct += float(len(correct_ind[0]))
                correct_close += float(len(correct_ind_close[0]))
            size += k
    per_class_acc = per_class_correct / per_class_num
    close_p = float(per_class_correct_cls.sum() / per_class_num.sum())
    print(
        '\nTest set including unknown classes:  Accuracy: {}/{} ({:.0f}%)  '
        '({:.4f}%)\n'.format(
            correct, size,
            100. * correct / size, float(per_class_acc.mean())))
    output = [step, list(per_class_acc), 'per class mean acc %s'%float(per_class_acc.mean()),
              float(correct / size), 'closed acc %s'%float(close_p)]
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=filename, format="%(message)s")
    logger.setLevel(logging.INFO)
    print(output)
    logger.info(output)


def test_class_inc(step, dataset_test, name, num_class, G, C, known_class):
    G.eval()
    C.eval()
    ## Known Score Calculation.
    correct = 0
    size = 0
    per_class_num = np.zeros((num_class))
    per_class_correct = np.zeros((num_class)).astype(np.float32)
    class_list = [i for i in range(num_class)]
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t, path_t = data[0], data[1], data[2]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            feat = G(img_t)
            out_t = C(feat)
            out_t = F.softmax(out_t)
            pred = out_t.data.max(1)[1]
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]

            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            label_t = label_t.data.cpu().numpy()
            if batch_idx == 0:
                label_all = label_t
                pred_all = out_t.data.cpu().numpy()
            else:
                pred_all = np.r_[pred_all, out_t.data.cpu().numpy()]
                label_all = np.r_[label_all, label_t]
    per_class_acc = per_class_correct / per_class_num
    print(
        '\nTest set including unknown classes:  Accuracy: {}/{} ({:.0f}%)  '
        '({:.4f}%)\n'.format(
            correct, size,
            100. * correct / size, float(per_class_acc.mean())))
    close_p = 100. * float(correct) / float(size)
    output = [step, "closed", list(per_class_acc), float(per_class_acc.mean()),
              "acc known %s"%float(per_class_acc[:known_class].mean()),
              "acc novel %s"%float(per_class_acc[known_class:].mean()), "acc %s"%float(close_p)]
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    logger.info(output)
    print(output)
    return float(per_class_acc[:known_class].mean()), float(per_class_acc[known_class:].mean())


def feat_get(step, G, C1, dataset_source, dataset_target, save_path):
    G.eval()
    C1.eval()

    for batch_idx, data in enumerate(dataset_source):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_s = data[0]
            label_s = data[1]
            img_s, label_s = Variable(img_s.cuda()), \
                             Variable(label_s.cuda())
            feat_s = G(img_s)
            if batch_idx == 0:
                feat_all_s = feat_s.data.cpu().numpy()
                label_all_s = label_s.data.cpu().numpy()
            else:
                feat_s = feat_s.data.cpu().numpy()
                label_s = label_s.data.cpu().numpy()
                feat_all_s = np.r_[feat_all_s, feat_s]
                label_all_s = np.r_[label_all_s, label_s]
    for batch_idx, data in enumerate(dataset_target):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_t = data[0]
            label_t = data[1]
            img_t, label_t = Variable(img_t.cuda()), \
                             Variable(label_t.cuda())
            feat_t = G(img_t)
            if batch_idx == 0:
                feat_all = feat_t.data.cpu().numpy()
                label_all = label_t.data.cpu().numpy()
            else:
                feat_t = feat_t.data.cpu().numpy()
                label_t = label_t.data.cpu().numpy()
                feat_all = np.r_[feat_all, feat_t]
                label_all = np.r_[label_all, label_t]
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "save_%s_target_feat.npy" % step), feat_all)
    np.save(os.path.join(save_path, "save_%s_source_feat.npy" % step), feat_all_s)
    np.save(os.path.join(save_path, "save_%s_target_label.npy" % step), label_all)
    np.save(os.path.join(save_path, "save_%s_source_label.npy" % step), label_all_s)
