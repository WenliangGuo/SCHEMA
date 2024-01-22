import torch
import os
import time
import numpy as np

from utils import *
from metrics import *
from torch.utils.data import DataLoader
from models.procedure_model import ProcedureModel
from models.helper import AverageMeter
from tensorboardX import SummaryWriter 

## add argument
from tools.parser import create_parser

# torch.autograd.set_detect_anomaly(True)

def eval(
        args,
        data_loader,
        model,
        logger,
        state_prompt_features,
        action_prompt_features,
        transition_matrix,
        e=0,
        device=torch.device("cuda"),
        writer=None,
        observation_features=None,
        is_train=False
    ):
    use_task = not args.no_task
    use_state_pred = not args.no_state_pred

    # losses
    losses_state  = AverageMeter()
    losses_action = AverageMeter()
    losses_action_desc = AverageMeter()
    if use_state_pred:
        losses_state_pred = AverageMeter()
    if use_task:
        losses_task = AverageMeter()

    # metrics for action
    action_acc1 = AverageMeter()
    action_acc5 = AverageMeter()
    action_sr   = AverageMeter()
    action_miou = AverageMeter()

    # metrics for viterbi
    viterbi_sr = AverageMeter()
    viterbi_acc1 = AverageMeter()
    viterbi_miou = AverageMeter()

    state_acc = AverageMeter()
    if use_task:
        task_acc = AverageMeter()

    with torch.no_grad():
        for i, (batch_states, batch_actions, batch_tasks) in enumerate(data_loader):
            '''
            batch_states:  (batch_size, time_horizon, 2, embedding_dim)
            batch_actions: (batch_size, time_horizon)
            batch_prompts: (batch_size, 2*time_horizon, num_prompts, embedding_dim)
            '''

            batch_size, _ = batch_actions.shape

            ## compute loss
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_tasks = batch_tasks.to(device)

            outputs, labels, losses = model(
                visual_features = batch_states,
                state_prompt_features = state_prompt_features,
                action_prompt_features = action_prompt_features,
                observation_features = observation_features,
                actions = batch_actions,
                tasks = batch_tasks,
                transition_matrix = transition_matrix
            )

            losses_state.update(losses["state_encode"].item(), batch_size)
            losses_action.update(losses["action"].item(), batch_size)
            losses_action_desc.update(losses["action_proj"].item(), batch_size)
            if use_state_pred:
                losses_state_pred.update(losses["state_decode"].item(), batch_size)
            if use_task:
                losses_task.update(losses["task"].item(), batch_size)

            ## metrics for state encoding
            acc_state = topk_accuracy(output=outputs["state_encode"].cpu(), target=labels["state"].cpu())
            state_acc.update(acc_state[0].item())

            ## computer accuracy for action prediction
            (acc1, acc5), sr, MIoU = \
                accuracy(outputs["action"].contiguous().view(-1, outputs["action"].shape[-1]).cpu(), 
                         labels["action"].contiguous().view(-1).cpu(), topk=(1, 5), max_traj_len=args.max_traj_len) 
            action_acc1.update(acc1.item(), batch_size)
            action_acc5.update(acc5.item(), batch_size)
            action_sr.update(sr.item(), batch_size)
            action_miou.update(MIoU, batch_size)

            # metrics for task prediction
            if use_task:
                acc_task = topk_accuracy(output=outputs["task"].cpu(), target=labels["task"].cpu(), topk=[1])[0]
                task_acc.update(acc_task.item(), batch_size)

            # viterbi decoding
            pred_viterbi = outputs["pred_viterbi"].cpu().numpy()
            labels_viterbi = labels["action"].reshape(batch_size, -1).cpu().numpy().astype("int")
            sr_viterbi = success_rate(pred_viterbi, labels_viterbi, True)
            miou_viterbi = acc_iou(pred_viterbi, labels_viterbi, False).mean()
            acc_viterbi = mean_category_acc(pred_viterbi, labels_viterbi)
            viterbi_sr.update(sr_viterbi, batch_size)
            viterbi_acc1.update(acc_viterbi, batch_size)
            viterbi_miou.update(miou_viterbi, batch_size)

        logger.info("Epoch: {} State Loss: {:.2f} Top1 Acc: {:.2f}%"\
                    .format(e+1, losses_state.avg, state_acc.avg))
        logger.info("\tAction Loss: {:.2f}, SR: {:.2f}% Acc1: {:.2f}% Acc5: {:.2f}% MIoU: {:.2f}"\
                    .format(losses_action.avg,
                            action_sr.avg,
                            action_acc1.avg,
                            action_acc5.avg,
                            action_miou.avg))
        logger.info("\tViterbi, SR: {:.2f}% Acc: {:.2f}% MIoU: {:.2f}"\
                    .format(viterbi_sr.avg,
                            viterbi_acc1.avg,
                            viterbi_miou.avg))
        if use_task:
            logger.info("\tTask Loss: {:.2f}, Acc1: {:.2f}%"\
                        .format(losses_task.avg, task_acc.avg))
        if use_state_pred:
            logger.info("\tState Pred Loss: {:.2f}"\
                        .format(losses_state_pred.avg))

        if is_train:
            writer.add_scalar('valid_loss/state', losses_state.avg, e+1)
            writer.add_scalar('valid_loss/action', losses_action.avg, e+1)
            writer.add_scalar('valid_loss/action_desc', losses_action_desc.avg, e+1)
            if use_task:
                writer.add_scalar('valid_loss/task', losses_task.avg, e+1)
            if use_state_pred:
                writer.add_scalar('valid_loss/state_pred', losses_state_pred.avg, e+1)

            writer.add_scalar('valid_state/acc', state_acc.avg, e+1)

            writer.add_scalar('valid_action/sr', action_sr.avg, e+1)
            writer.add_scalar('valid_action/miou', action_miou.avg, e+1)
            writer.add_scalar('valid_action/acc1', action_acc1.avg, e+1)
            writer.add_scalar('valid_action/acc5', action_acc5.avg, e+1)

            writer.add_scalar('valid_action/viterbi_sr', viterbi_sr.avg, e+1)
            writer.add_scalar('valid_action/viterbi_miou', viterbi_miou.avg, e+1)
            writer.add_scalar('valid_action/viterbi_acc1', viterbi_acc1.avg, e+1)

            if use_task:
                writer.add_scalar('valid_task/acc', task_acc.avg, e+1)

    return viterbi_sr.avg


def evaluate(args):
    log_file_path = os.path.join(args.saved_path, f"{args.model_name}_T{args.max_traj_len}_log_eval.txt")
    logger = get_logger(log_file_path)
    logger.info("{}".format(log_file_path))
    logger.info("{}".format(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'crosstask':
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/crosstask_state_prompt_features.npy')
        action_prompt_features = np.load('./data/action_description_features/crosstask_action_prompt_features.npy')

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
        # observation_features = train_dataset.state_features
    
    elif args.dataset == "coin":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/coin_state_prompt_features.npy')
        action_prompt_features = np.load('./data/action_description_features/coin_action_prompt_features.npy')
    
        logger.info("Loading training data...")
        train_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.train_json, args.max_traj_len, aug_range=args.aug_range, 
                                        mode = "train", M=args.M)
        logger.info("Loading valid data...")
        valid_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.valid_json, args.max_traj_len, aug_range=args.aug_range, 
                                        mode = "valid", M=args.M)
        transition_matrix = train_dataset.transition_matrix
        # observation_features = train_dataset.state_features

    elif args.dataset == "niv":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/niv_state_prompt_features.npy')
        action_prompt_features = np.load('./data/action_description_features/niv_action_prompt_features.npy')

        logger.info("Loading training data...")
        train_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.train_json, args.max_traj_len, num_action = 48,
                                        aug_range=args.aug_range, mode = "train", M=args.M)
        
        logger.info("Loading valid data...")
        valid_dataset = ProcedureDataset(args.features_dir, state_prompt_features,
                                        args.valid_json, args.max_traj_len, num_action = 48,
                                        aug_range=args.aug_range, mode = "valid", M=args.M)
        transition_matrix = train_dataset.transition_matrix
        # observation_features = train_dataset.state_features
    
    logger.info("Training set volumn: {} Testing set volumn: {}".format(len(train_dataset), len(valid_dataset)))
    
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = ProcedureModel(
        vis_input_dim=args.img_input_dim,
        lang_input_dim=args.text_input_dim,
        embed_dim=args.embed_dim,
        time_horz=args.max_traj_len, 
        num_classes=args.num_action,
        num_tasks=args.num_tasks,
        args=args
    ).to(device)

    model_path = os.path.join(args.saved_path, f'{args.model_name}_T{args.max_traj_len}.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    state_prompt_features  = torch.tensor(state_prompt_features).to(device, dtype=torch.float32).clone().detach()
    action_prompt_features = torch.tensor(action_prompt_features).to(device, dtype=torch.float32).clone().detach()
    eval(
        args,
        valid_loader, 
        model,
        logger, 
        state_prompt_features, 
        action_prompt_features,
        transition_matrix, 
        -1,
        device
    )


def train(args):
    use_task = not args.no_task
    use_state_pred = not args.no_state_pred
    use_state_pred_loss = not args.no_state_decode_loss
    use_state_encode_loss = not args.no_state_encode_loss
    use_action_proj_loss = not args.no_action_proj_loss
    # use_state_recon_loss = args.use_state_recon_loss

    path = "logs/{}_{}_len{}_seed{}".format(
        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), 
        args.model_name, 
        args.max_traj_len,
        args.seed)
    if args.last_epoch > -1:
        path += "_last{}".format(args.last_epoch)
    args.saved_path = path
    os.makedirs(path)
    log_file_path = os.path.join(path, "log.txt")
    logger = get_logger(log_file_path)
    logger.info("{}".format(log_file_path))
    logger.info("{}".format(args))

    validate_interval = 1
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'crosstask':
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/crosstask_state_prompt_features.npy')
        action_prompt_features = np.load('./data/action_description_features/crosstask_action_prompt_features.npy')

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
        # observation_features = train_dataset.state_features
    
    elif args.dataset == "coin":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/coin_state_prompt_features.npy')
        action_prompt_features = np.load('./data/action_description_features/coin_action_prompt_features.npy')
    
        logger.info("Loading training data...")
        train_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.train_json, args.max_traj_len, aug_range=args.aug_range, 
                                        mode = "train", M=args.M)
        logger.info("Loading valid data...")
        valid_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.valid_json, args.max_traj_len, aug_range=args.aug_range, 
                                        mode = "valid", M=args.M)
        transition_matrix = train_dataset.transition_matrix
        # observation_features = train_dataset.state_features

    elif args.dataset == "niv":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/niv_state_prompt_features.npy')
        action_prompt_features = np.load('./data/action_description_features/niv_action_prompt_features.npy')

        logger.info("Loading training data...")
        train_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.train_json, args.max_traj_len, num_action = 48,
                                        aug_range=args.aug_range, mode = "train", M=args.M)
        
        logger.info("Loading valid data...")
        valid_dataset = ProcedureDataset(args.features_dir, state_prompt_features,
                                        args.valid_json, args.max_traj_len, num_action = 48,
                                        aug_range=args.aug_range, mode = "valid", M=args.M)
        transition_matrix = train_dataset.transition_matrix
        # observation_features = train_dataset.state_features

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info("Training set volumn: {} Testing set volumn: {}".format(len(train_dataset), len(valid_dataset)))

    writer = SummaryWriter(path)

    ## begin training
    model = ProcedureModel(
        vis_input_dim=args.img_input_dim,
        lang_input_dim=args.text_input_dim,
        embed_dim=args.embed_dim,
        time_horz=args.max_traj_len, 
        num_classes=args.num_action,
        num_tasks=args.num_tasks,
        args=args
    ).to(device)

    # logger.info("Step decoder use_random: {}".format(model.state_decoder.use_random))
    # logger.info("Action decoder use_random: {}".format(model.action_decoder.use_random))

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters()},
        ],
        lr=args.lr
    )

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args.step_size, 
        gamma=args.lr_decay, 
        last_epoch=-1
    )

    state_prompt_features  = torch.tensor(state_prompt_features).to(device, dtype=torch.float32).clone().detach()
    action_prompt_features = torch.tensor(action_prompt_features).to(device, dtype=torch.float32).clone().detach()
    # observation_features = torch.tensor(observation_features).to(device, dtype=torch.float32).clone().detach()
    observation_features = None

    max_SR = 0

    for e in range(0, args.epochs):
        model.train()

        # losses
        losses_state  = AverageMeter()
        losses_action = AverageMeter()
        losses_action_desc = AverageMeter()
        if use_task:
            losses_task = AverageMeter()
        if use_state_pred:
            losses_state_pred = AverageMeter()

        # metrics for action
        action_acc1 = AverageMeter()
        action_acc5 = AverageMeter()
        action_sr   = AverageMeter()
        action_miou = AverageMeter()
        state_acc = AverageMeter()
        if use_task:
            task_acc = AverageMeter()

        for i, (batch_states, batch_actions, batch_tasks) in enumerate(train_loader):
            '''
            batch_states:  (batch_size, time_horizon, 2, embedding_dim)
            batch_actions: (batch_size, time_horizon)
            '''
            batch_size, _ = batch_actions.shape
            optimizer.zero_grad()

            ## compute loss
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_tasks = batch_tasks.to(device)
            outputs, labels, losses = model(
                visual_features=batch_states,
                state_prompt_features=state_prompt_features,
                action_prompt_features=action_prompt_features,
                observation_features = observation_features,
                actions=batch_actions,
                tasks=batch_tasks
            )

            total_loss = losses["action"]
            if use_action_proj_loss:
                total_loss += losses["action_proj"] * 0.1
            if use_state_encode_loss:
                total_loss += losses["state_encode"]
            if use_task:
                total_loss += losses["task"]
            if use_state_pred & use_state_pred_loss:
                total_loss += losses["state_decode"] * 0.1
                # if use_state_recon_loss:
                #     total_loss += losses["state_recon"] * 0.1

            total_loss.backward()
            optimizer.step()

            losses_state.update(losses["state_encode"].item())
            losses_action.update(losses["action"].item())
            losses_action_desc.update(losses["action_proj"].item())
            if use_state_pred:
                losses_state_pred.update(losses["state_decode"].item())
            if use_task:
                losses_task.update(losses["task"].item())

            ## computer accuracy for state encoding
            acc_state = topk_accuracy(output=outputs["state_encode"].cpu(), target=labels["state"].cpu())
            state_acc.update(acc_state[0].item())

            ## computer accuracy for action prediction
            (acc1, acc5), sr, MIoU = \
                accuracy(outputs["action"].contiguous().view(-1, outputs["action"].shape[-1]).cpu(), 
                         labels["action"].contiguous().view(-1).cpu(), topk=(1, 5), max_traj_len=args.max_traj_len) 
            action_acc1.update(acc1.item())
            action_acc5.update(acc5.item())
            action_sr.update(sr.item())
            action_miou.update(MIoU)

            if use_task:
                acc_task = topk_accuracy(output=outputs["task"].cpu(), target=labels["task"].cpu(), topk=[1])[0]
                task_acc.update(acc_task.item())

        logger.info("Epoch: {} State Loss: {:.2f} Top1 Acc: {:.2f}%"\
                    .format(e+1, losses_state.avg, state_acc.avg))
        logger.info("\tAction Loss: {:.2f}, SR: {:.2f}% Acc1: {:.2f}% Acc5: {:.2f}% MIoU: {:.2f}"\
                    .format(losses_action.avg,
                            action_sr.avg,
                            action_acc1.avg,
                            action_acc5.avg,
                            action_miou.avg))
        if use_task:
            logger.info("\tTask Loss: {:.2f}, Acc1: {:.2f}%"\
                        .format(losses_task.avg, task_acc.avg))
        if use_state_pred:
            logger.info("\tState Pred Loss: {:.2f}"\
                        .format(losses_state_pred.avg))

        # lr_state_enc = optimizer_state_enc.param_groups[0]['lr']
        # lr_action = optimizer_action.param_groups[0]['lr']
        # writer.add_scalar('lr/state_enc', lr_state_enc, e+1)
        # writer.add_scalar('lr/action', lr_action, e+1)

        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr/lr', lr, e+1)

        writer.add_scalar('train_loss/state', losses_state.avg, e+1)
        writer.add_scalar('train_loss/action', losses_action.avg, e+1)
        writer.add_scalar('train_loss/action_desc', losses_action_desc.avg, e+1)
        if use_task:
            # lr_task = optimizer_task.param_groups[0]['lr']
            # writer.add_scalar('lr/task', lr_task, e+1)
            writer.add_scalar('train_loss/task', losses_task.avg, e+1)
        if use_state_pred:
            # lr_state_dec = optimizer_state_dec.param_groups[0]['lr']
            # writer.add_scalar('lr/state_dec', lr_state_dec, e+1)
            writer.add_scalar('train_loss/state_pred', losses_state_pred.avg, e+1)

        writer.add_scalar('train_state/acc', state_acc.avg, e+1)

        writer.add_scalar('train_action/sr', action_sr.avg, e+1)
        writer.add_scalar('train_action/miou', action_miou.avg, e+1)
        writer.add_scalar('train_action/acc1', action_acc1.avg, e+1)
        writer.add_scalar('train_action/acc5', action_acc5.avg, e+1)

        if use_task:
            writer.add_scalar('train_task/acc', task_acc.avg, e+1)

        if args.last_epoch < 0 or e < args.last_epoch:
            scheduler.step()

        ## validation
        if (e+1)%validate_interval == 0:
            model.eval()
            SR = eval(args, 
                      valid_loader, 
                      model, 
                      logger, 
                      state_prompt_features, 
                      action_prompt_features, 
                      transition_matrix, 
                      e, 
                      device,
                      writer=writer, 
                      is_train=True)
            
            torch.save(
                model.state_dict(), 
                os.path.join(
                    args.saved_path, 
                    "model_last.pth"
                )
            )

            if SR > max_SR:
                max_SR = SR
                torch.save(
                    model.state_dict(), 
                    os.path.join(
                        args.saved_path, 
                        "model_best.pth"
                    )
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

    if args.eval:
        evaluate(args)
    else:
        train(args)