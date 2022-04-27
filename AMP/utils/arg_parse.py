import argparse
import torch


def get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='debug')
    parser.add_argument('--exp_name', default="none", type=str)
    args = parser.parse_args()
    return args