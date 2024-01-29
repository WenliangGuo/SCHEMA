import torch
import torch.nn as nn
import os
import numpy as np

from utils import *
from metrics import *
from torch.utils.data import DataLoader
from models.procedure_model import ProcedureModel
from models.utils import AverageMeter
from models.utils import viterbi_path
from tools.parser import create_parser

## implementation based on https://github.dev/MCG-NJU/PDPP
def cal_uncertainty(actions_pred_logits, gt, num_sampling, horizon, act_size, all_ref):
    actions_pred = torch.argmax(actions_pred_logits, dim=-1)
    actions_pred = actions_pred.view(num_sampling, -1)
    sample_listing = actions_pred

    bz = all_ref.shape[0]
    gt_sample = np.repeat(gt.cpu().numpy(), bz, axis=0)

    criter = (
        (gt_sample[:, [0, -1]] == all_ref[:, [0, -1]])
            .all(axis=1)
            .nonzero()[0]
    )

    dist_samples = all_ref[criter]
    len_unique = len(np.unique(dist_samples, axis=0))
    ref_onehot = torch.FloatTensor(horizon, act_size).cuda()
    ref_onehot.zero_()

    ######################################################################
    # dist_samples represents the samples in the test-set:               #
    #    1). Share the same start and end-goal semantic;                 #
    #                                                                    #
    # If can not find any dist_samples (aka dist_samples.shape[0] == 0): #
    #    1). Skip the nll evaluation (see below code)                    #
    ######################################################################
    if dist_samples.shape[0] != 0:
        for vec in dist_samples:
            vec = torch.from_numpy(vec).cuda()
            ref_onehot_tmp = torch.FloatTensor(
                horizon, act_size
            ).cuda()
            ref_onehot_tmp.zero_()
            ref_onehot_tmp.scatter_(
                1, vec.view(horizon, -1), 1)
            ref_onehot += ref_onehot_tmp

        ref_dist = ref_onehot
        itm_onehot = torch.FloatTensor(horizon, act_size).cuda()
        itm_onehot.zero_()

        for itm in sample_listing:
            ###########################################
            # Convert indivisual sample into onehot() #
            ###########################################
            itm_onehot_tmp = torch.FloatTensor(horizon, act_size).cuda()
            itm_onehot_tmp.zero_()
            itm_onehot_tmp.scatter_(
                1, itm.cuda().view(horizon, -1), 1)
            itm_onehot += itm_onehot_tmp

    ###########################################
    # Evaluate on Mode-Coverage Prec & Recall #
    ###########################################
    ratio_list = []
    for sample in sample_listing:
        ratio_list.append(
            (sample.squeeze().cpu().numpy() == dist_samples).all(1).any()
        )
    ratio = sum(ratio_list) / num_sampling
    mc_prec = ratio

    # all_samples = torch.stack(
    #     sample_listing).squeeze().cpu().numpy()
    all_samples = sample_listing.cpu().numpy()

    # dist_samples_unique = np.unique(dist_samples, axis=0)
    dist_samples_unique = dist_samples
    num_expert = dist_samples_unique.shape[0]
    list_expert = np.array_split(dist_samples_unique, num_expert)
    tmp_recall = []
    for item in list_expert:
        tmp_recall.append((item == all_samples).all(1).any())
    mc_recall = sum(tmp_recall) / len(tmp_recall)

    ####################################
    #   Calculate the KL-Div  Metric   #
    ####################################

    ref_dist /= dist_samples.shape[0]
    itm_onehot /= num_sampling
    ref_dist *= 10
    itm_onehot *= 10

    ref_dist = ref_dist.softmax(dim=-1)
    itm_onehot = itm_onehot.softmax(dim=-1)

    klv_rst = (
        torch.nn.functional.kl_div(
            itm_onehot.log(),
            ref_dist,
            reduction='batchmean'
        )
        .cpu()
        .numpy()
    )
    klv_rst = np.where(np.isnan(klv_rst), 0, klv_rst)
    klv_rst = np.where(np.isinf(klv_rst), 0, klv_rst)
    klv = klv_rst

    ####################################
    #   Calculate the NLL  Metric   #
    ####################################

    klv_rst2 = (
        torch.mean(-torch.sum(ref_dist * itm_onehot.log(), 1)).cpu().numpy()
    )
    klv_rst2 = np.where(np.isnan(klv_rst2), 0, klv_rst2)
    klv_rst2 = np.where(np.isinf(klv_rst2), 0, klv_rst2)
    nll = klv_rst2

    return len_unique, mc_prec, mc_recall, klv, nll


## implementation based on https://github.dev/JoeHEZHAO/procedure-planing
def cal_viterbi(rst_argmax, act_size, pred_horz, num_sampling, transition_matrix):
    # """Formulate distribution from these samples, for viterbi results """
    ref_onehot = torch.FloatTensor(pred_horz, act_size).cuda()
    ref_onehot.zero_()

    """Make this run in parallel"""
    ref_onehot_tmp = torch.FloatTensor(rst_argmax.shape[0], 
                                       pred_horz, 
                                       act_size).cuda().zero_()   # [num_sample, T, num_act]
    
    ref_onehot_tmp.scatter_(2, rst_argmax.view(rst_argmax.shape[0], pred_horz, -1), 1)
    ref_onehot = ref_onehot_tmp.sum(0)

    "Normalize with total number of samples"
    new_logits = ref_onehot / num_sampling

    #################
    #  Run Viterbi  #
    #################
    viterbi_rst = viterbi_path(
        transition_matrix,
        new_logits.permute(1, 0).cpu().numpy()
    )

    return viterbi_rst

def eval(
        args,
        data_loader,
        model,
        logger,
        state_prompt_features,
        transition_matrix,
        e=0,
        device=torch.device("cuda"),
        writer=None,
        is_train=False
    ):

    # metrics for action
    action_acc1 = AverageMeter()
    action_acc5 = AverageMeter()
    action_sr   = AverageMeter()
    action_miou = AverageMeter()

    # metrics for uncertainty
    uncertain_len_unique = AverageMeter()
    uncertain_mc_prec = AverageMeter()
    uncertain_mc_recall = AverageMeter()
    uncertain_kl = AverageMeter()
    uncertain_nll = AverageMeter()

    # metrics for viterbi
    viterbi_sr = AverageMeter()
    viterbi_acc1 = AverageMeter()
    viterbi_miou = AverageMeter()

    reference = []
    for i, (batch_states, batch_actions, batch_tasks) in enumerate(data_loader):
        reference.append(batch_actions.cpu().numpy())
    all_ref = np.concatenate(reference, axis=0)   # [num_valid, time_horizon]

    with torch.no_grad():
        for i, (batch_states, batch_actions, batch_tasks) in enumerate(data_loader):
            '''
            batch_states:   (bs, time_horizon, 2, embedding_dim)
            batch_actions:  (bs, time_horizon)
            batch_tasks:    (bs)
            '''
            temp_outputs = {"action": [], "viterbi": []}
            temp_labels = {"action": [], "viterbi": []}

            for j in range(batch_states.shape[0]):
                batch_size, _ = batch_actions.shape

                input_states = batch_states[j:j+1].repeat(args.num_sample, 1, 1, 1).to(device)      # [num_sample, time_horizon, 2, embed_dim]
                input_actions = batch_actions[j].unsqueeze(0).repeat(args.num_sample, 1).to(device)     # [num_sample, time_horizon]
                input_tasks = batch_tasks[j].repeat(args.num_sample).to(device)     # [num_sample]

                outputs, labels, losses = model(
                    visual_features = input_states,
                    state_prompt_features = state_prompt_features,
                    observation_features = None,
                    actions = input_actions,
                    tasks = input_tasks,
                    transition_matrix = transition_matrix
                )

                action_logits = outputs["action"].reshape(-1, args.max_traj_len, args.num_action)   # [num_sample, time_horizon, num_action]
                pred_action = action_logits.argmax(-1)   # [num_sample, time_horizon]
                temp_outputs["action"].append(pred_action[0])
                temp_labels["action"].append(batch_actions[j])

                ## Viterbi decoding
                viterbi_rst = cal_viterbi(pred_action, args.num_action, args.max_traj_len, args.num_sample, transition_matrix)
                temp_outputs["viterbi"].append(torch.from_numpy(viterbi_rst).cuda())
                temp_labels["viterbi"].append(batch_actions[j])

                ## Uncertainty metrics
                len_unique, mc_prec, mc_recall, klv, nll = \
                    cal_uncertainty(action_logits, 
                                    batch_actions[j].unsqueeze(0), 
                                    args.num_sample, 
                                    args.max_traj_len, 
                                    args.num_action, 
                                    all_ref)
                
                uncertain_len_unique.update(len_unique)
                uncertain_mc_prec.update(mc_prec)
                uncertain_mc_recall.update(mc_recall)
                uncertain_kl.update(klv)
                uncertain_nll.update(nll)

            ## action metrics
            temp_outputs["action"] = torch.stack(temp_outputs["action"], dim=0).cpu().numpy()   # [bs, time_horizon]
            temp_labels["action"] = torch.stack(temp_labels["action"], dim=0).cpu().numpy()     # [bs, time_horizon]
            sr = success_rate(temp_outputs["action"], temp_labels["action"], True)
            miou = acc_iou(temp_outputs["action"], temp_labels["action"], False).mean()
            acc = mean_category_acc(temp_outputs["action"], temp_labels["action"])
            action_acc1.update(acc, batch_size)
            action_sr.update(sr, batch_size)
            action_miou.update(miou, batch_size)

            # viterbi decoding metrics
            viterbi_pred = torch.stack(temp_outputs["viterbi"], dim=0).cpu().numpy()
            labels_viterbi = torch.stack(temp_labels["viterbi"], dim=0).cpu().numpy()     # [bs, time_horizon]
            sr_viterbi = success_rate(viterbi_pred, labels_viterbi, True)
            miou_viterbi = acc_iou(viterbi_pred, labels_viterbi, False).mean()
            acc_viterbi = mean_category_acc(viterbi_pred, labels_viterbi)
            viterbi_sr.update(sr_viterbi, batch_size)
            viterbi_acc1.update(acc_viterbi, batch_size)
            viterbi_miou.update(miou_viterbi, batch_size)

    logger.info("\tAction, SR: {:.2f}% Acc: {:.2f}% MIoU: {:.2f}"\
                .format(action_sr.avg,
                        action_acc1.avg,
                        action_miou.avg))
    logger.info("\tViterbi, SR: {:.2f}% Acc: {:.2f}% MIoU: {:.2f}"\
                .format(viterbi_sr.avg,
                        viterbi_acc1.avg,
                        viterbi_miou.avg))
    logger.info("\tUncertainty, Len_unique: {:.2f} MC_prec: {:.2f} MC_recall: {:.2f} KL: {:.2f} NLL: {:.2f}"\
                .format(uncertain_len_unique.avg,
                        uncertain_mc_prec.avg*100,
                        uncertain_mc_recall.avg*100,
                        uncertain_kl.avg,
                        uncertain_nll.avg))


def main_worker(args):
    log_file_path = os.path.join(args.saved_path, f"uncertain_{args.model_name}_T{args.max_traj_len}_log_eval.txt")
    logger = get_logger(log_file_path)
    logger.info("{}".format(log_file_path))
    logger.info("{}".format(args))

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'crosstask':
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/crosstask_state_prompt_features.npy')

        ## parse raw data
        task_info_path = os.path.join(args.root_dir, "tasks_primary.txt")
        task_info = parse_task_info(task_info_path)
        with open("data/crosstask_idices.json", "r") as f:
            idices_mapping = json.load(f)
        anot_dir = os.path.join(args.root_dir, "annotations")
        anot_info = parse_annotation(anot_dir, task_info, idices_mapping)

        ## create procedure datset and dataloader
        logger.info("Loading training data...")
        train_dataset = ProcedureDataset(anot_info, args.features_dir, state_prompt_features, 
                                        args.train_json, args.max_traj_len, aug_range=args.aug_range, 
                                        dataset=args.dataset, datasplit=args.split, mode = "train", M=args.M)
        logger.info("Loading valid data...")
        valid_dataset = ProcedureDataset(anot_info, args.features_dir, state_prompt_features, 
                                        args.valid_json, args.max_traj_len, aug_range=args.aug_range, 
                                        dataset=args.dataset, datasplit=args.split, mode = "valid", M=args.M)
        transition_matrix = train_dataset.transition_matrix
    
    elif args.dataset == "coin":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/coin_state_prompt_features.npy')

        logger.info("Loading training data...")
        train_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.train_json, args.max_traj_len, aug_range=args.aug_range, 
                                        mode = "train", M=args.M)
        
        logger.info("Loading valid data...")
        valid_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.valid_json, args.max_traj_len, aug_range=args.aug_range, 
                                        mode = "valid", M=args.M)
        transition_matrix = train_dataset.transition_matrix

    elif args.dataset == "niv":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/niv_state_prompt_features.npy')

        logger.info("Loading training data...")
        train_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.train_json, args.max_traj_len, num_action = 48,
                                        aug_range=args.aug_range, mode = "train", M=args.M)
        
        logger.info("Loading valid data...")
        valid_dataset = ProcedureDataset(args.features_dir, state_prompt_features,
                                        args.valid_json, args.max_traj_len, num_action = 48,
                                        aug_range=args.aug_range, mode = "valid", M=args.M)
        transition_matrix = train_dataset.transition_matrix
    
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logger.info("Training set volumn: {} Testing set volumn: {}".format(len(train_dataset), len(valid_dataset)))
    
    model = ProcedureModel(
        vis_input_dim=args.img_input_dim,
        lang_input_dim=args.text_input_dim,
        embed_dim=args.embed_dim,
        time_horz=args.max_traj_len, 
        num_classes=args.num_action,
        num_tasks=args.num_tasks,
        args=args
    ).to(device)

    model_path = os.path.join(args.saved_path, f'uncertain_{args.model_name}_T{args.max_traj_len}.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    state_prompt_features  = torch.tensor(state_prompt_features).to(device, dtype=torch.float32).clone().detach()
    eval(
        args,
        valid_loader, 
        model,
        logger, 
        state_prompt_features, 
        transition_matrix, 
        -1,
        device
    )


if __name__ == "__main__":
    args = create_parser()

    if args.dataset == 'crosstask':
        if args.split == 'base' or args.split == 'pdpp':
            from dataset.crosstask_dataloader import CrossTaskDataset as ProcedureDataset
            # ## use pdpp data-sample
            # from dataset.crosstask_dataloader_pdpp import CrossTaskDataset as ProcedureDataset
        elif args.split == 'p3iv':
            from dataset.crosstask_dataloader_p3iv import CrossTaskDataset as ProcedureDataset
    
    elif args.dataset == 'coin':
        from dataset.coin_dataloader import CoinDataset as ProcedureDataset
    
    elif args.dataset == 'niv':
        from dataset.niv_dataloader import NivDataset as ProcedureDataset


    main_worker(args)