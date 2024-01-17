import argparse

## add argument
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
                        default='model', type=str, 
                        help='model name')
    parser.add_argument('--num_layers', 
                        default=3, type=int, metavar='NUM_LAYERS',
                        help='number of layers (default: 3)')
    parser.add_argument('--attn_heads', 
                        default=4, type=int, metavar='NUM_HEADS',
                        help='number of heads (default: 4)')
    parser.add_argument('--mlp_ratio', 
                        default=2, type=int, metavar='MLP_RATIO',
                        help='mlp ratio in ff (default: 2)')
    # parser.add_argument('--input_dim', 
    #                     default=768, type=int, metavar='DIM',
    #                     help='dimension (default: 768)')
    parser.add_argument('--embed_dim', 
                        default=128, type=int, metavar='DIM',
                        help='dimension (default: 128)')

    parser.add_argument('--fully_supervised', 
                        action='store_true',
                        help='Align mid-state with corresponding descriptions')
    parser.add_argument('--distributed', 
                        action='store_true',
                        help='distributed training')
    parser.add_argument('--sample_frame', 
                        action='store_true',
                        help='sample frame')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--max_traj_len', 
                        default=3, type=int, metavar='MAXTRAJ',
                        help='max length (default: 54)')
    parser.add_argument('--epochs', 
                        default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', '-b', 
                        default=72, type=int,
                        metavar='N', help='mini-batch size (default: 72)')
    parser.add_argument('--dropout',
                        default=0.1, type=float,
                        help='dropout rate')

    parser.add_argument('--optimizer', 
                        default='adam', type=str, 
                        help='optimizer (default: sgd)')
    parser.add_argument('--lr', '--learning-rate', 
                        default=0.01, type=float, metavar='LR', 
                        help='initial learning rate')
    parser.add_argument('--step_size', 
                        default=20, type=int, metavar='LRSteps', 
                        help='epochs to decay learning rate')
    parser.add_argument('--lr_decay',
                        default=0.65, type=float,
                        help='learning weight decay')
    
    parser.add_argument('--momentum', 
                        default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--queue_size', 
                        default=8192, type=int, metavar='M',
                        help='queue size')
    parser.add_argument('--weight_decay', '--wd', 
                        default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--M',
                        default=1, type=int,
                        metavar='W', help='augmentation factor (default: 1)')
    parser.add_argument('--aug_range',
                        default=0, type=int,
                        metavar='W', help='augmentation range (default: 0)')

    parser.add_argument('--root_dir', 
                        default='/home/yulei/data/crosstask/crosstask_release', type=str, 
                        help='root dir')
    parser.add_argument('--train_json', 
                        default='/home/wenliang/procedure_planning/cross_task_data_False.json', type=str, 
                        help='train json file')
    parser.add_argument('--valid_json', 
                        default='/home/wenliang/procedure_planning/cross_task_data_True.json', type=str, 
                        help='valid json file')
    parser.add_argument('--features_dir', 
                        default='/home/yulei/data/crosstask/crosstask_features_clip_336px', type=str, 
                        help='features dir')
    parser.add_argument('--dsp_dir_json', 
                        default='/home/yulei/data/crosstask/gpt3_output_detail_v2_num3.json', type=str, 
                        help='descriptions dir')

    parser.add_argument('--eval', 
                        action='store_true',
                        help='evaluation mode')
    parser.add_argument('--saved_path', 
                        default='./logs/', type=str, 
                        help='descriptions dir')

    parser.add_argument('--last_epoch',
                        default=-1, type=int,
                        help='last epoch for adjusting learning rate')

    parser.add_argument('--split', 
                        default='base', type=str, 
                        help='split (base, p3iv)')

    parser.add_argument('--no_task', 
                        action='store_true',
                        help='not using task')
    parser.add_argument('--no_state_pred', 
                        action='store_true',
                        help='not using state decoder')
    parser.add_argument('--no_state_decode_loss', 
                        action='store_true',
                        help='not using state decode loss')
    parser.add_argument('--no_state_encode_loss', 
                        action='store_true',
                        help='not using state encode loss')
    parser.add_argument('--no_action_proj_loss', 
                        action='store_true',
                        help='not using action proj loss')
    parser.add_argument('--no_state_memory', 
                        action='store_true',
                        help='not using state memory')
    parser.add_argument('--use_action_memory', 
                        action='store_true',
                        help='use action memory')
    
    parser.add_argument('--use_random', 
                        action='store_true',
                        help='use random noise')
    
    parser.add_argument('--use_ensemble', 
                        action='store_true',
                        help='ensemble 3 outputs')
    
    parser.add_argument('--no_state_decoder', 
                        action='store_true',
                        help='not using state decoder')

    parser.add_argument('--seed', 
                        default=3407, type=int, metavar='M',
                        help='queue size')
    
    ## newly added 09/22
    parser.add_argument('--dataset', 
                        default='crosstask_howto100m', type=str, 
                        help='features')
    parser.add_argument('--text_input_dim', 
                        default=768, type=int, metavar='DIM',
                        help='dimension (default: 768)')
    parser.add_argument('--img_input_dim', 
                        default=768, type=int, metavar='DIM',
                        help='dimension (default: 512)')
    parser.add_argument('--description_type',
                        default='cot', type=str, 
                        help='')
    parser.add_argument('--num_action',
                        default=133, type=int,
                        help='number of action classes (crosstask: 133, coin: 778)')
    parser.add_argument('--num_tasks',
                        default=18, type=int,
                        help='number of tasks (crosstask: 18, coin: 778)')
    parser.add_argument('--use_observ_memory', 
                        action='store_true',
                        help='use training observations as memory')
    parser.add_argument('--use_state_recon_loss', 
                        action='store_true',
                        help='using state recon loss')
    parser.add_argument('--uncertain', 
                        action='store_true',
                        help='probabilistic model')
    parser.add_argument('--num_sample',
                         default=1500, type=int,
                         help='number of samples of noise-vectors')

    return parser.parse_args()