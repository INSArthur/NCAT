#!/usr/bin/python
# encoding: utf-8

import numpy as np
import ipdb
import logging
from datetime import datetime
import sys
from util import get_objects,set_global_seeds,arg_parser
import envs as all_envs
import agents as all_agents
import function as all_FA
import os
def str2bool(str=""):
    str = str.lower()
    if str.__contains__("yes") or str.__contains__("true") or str.__contains__("y") or str.__contains__("t"):
        return True
    else:
        return False

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run.py.
    """
    parser = arg_parser()
    parser.add_argument('-seed',type=int, default=123)
    parser.add_argument('-environment', type=str, default="Env")
    parser.add_argument('-data_path',type=str,default="./data/")
    parser.add_argument('-data_name',type=str,default="name")
    parser.add_argument('-agent',type=str,default="training methods")
    parser.add_argument('-FA',type=str,default="function approximation")
    parser.add_argument('-CDM', dest='CDM', type=str, default='CDM', help="type of CDM")
    parser.add_argument('-T', dest='T', type=int, default=3, help="time_step")
    parser.add_argument('-ST', dest='ST', type=eval, default="[10,30,60,120]", help="evaluation_time_step")
    parser.add_argument('-gpu_no', dest='gpu_no', type=str, default="0", help='which gpu for usage')
    parser.add_argument('-latent_factor', dest='latent_factor', type=int, default=10, help="latent factor")
    parser.add_argument('-learning_rate', dest='learning_rate', type=float, default=0.01, help="learning rate")
    parser.add_argument('-training_epoch', dest='training_epoch', type=int, default=30000, help="training epoch")
    parser.add_argument('-rnn_layer', dest='rnn_layer', type=int, default=1, help="rnn_layer")
    parser.add_argument('-inner_epoch', dest='inner_epoch', type=int, default=50, help="rnn_layer")
    parser.add_argument('-batch', dest='batch', type=int, default=128, help="batch_size")
    parser.add_argument('-gamma', dest='gamma', type=float, default=0.0, help="gamma")
    parser.add_argument('-clip_param', dest='clip_param', type=float, default=0.2, help="clip_param")
    parser.add_argument('-restore_model', dest='restore_model', type=str2bool, default="False", help="")
    parser.add_argument('-num_blocks', dest='num_blocks', type=int, default=1, help="")
    parser.add_argument('-num_heads', dest='num_heads', type=int, default=1, help="")
    parser.add_argument('-dropout_rate', dest='dropout_rate', type=float, default=0.0, help="")
    return parser

def main(args):
    # arguments
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    args.model = "_".join([args.agent,args.FA,str(args.T)])
    # initialization
    set_global_seeds(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)
    # logger
    logger = logging.getLogger()
    logger.info("./log"+args.data_path.split("/")[-2]+"/"+"_".join([args.model,datetime.now().strftime("%Y%m%d_%H%M%S"),args.data_path.split("/")[-2],str(args.learning_rate),str(args.T),str(args.ST),str(args.gamma)]))
    logger.info("Training Model: "+args.model)
    # environments
    envs = get_objects(all_envs)
    env = envs[args.environment](args)
    # policy network
    args.user_num = env.user_num
    args.item_num = env.item_num
    args.utype_num = env.utype_num
    args.saved_path = os.path.join(os.path.abspath("./"),"saved_path_"+args.data_path.split("/")[-2]+"_"+str(args.FA)+"_"+str(args.learning_rate)+"_"+str(args.agent)+"_"+str(args.seed))
    nets = get_objects(all_FA)
    fa = nets[args.FA].create_model(args)
    # return
        #TODO : fix this !
        #logger.log("Hype-Parameters: "+str(args))
    print(str(args))
    # # agents
    agents = get_objects(all_agents)
    agents[args.agent](env, fa, args).train()




if __name__ == '__main__':
    main(sys.argv)