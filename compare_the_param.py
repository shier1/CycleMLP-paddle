from genericpath import isfile
import os
import argparse
import paddle

parser = argparse.ArgumentParser("test the model param")
parser.add_argument('-model_path', type=str, default='output/best_cyclemlp301-350/Best_CycleMLP',help='the model path')
arguments = parser.parse_args()


def main():
    assert os.path.isfile(arguments.model_path+".pdparams") is True
    param_state_dict = paddle.load(arguments.model_path+".pdparams")
    for name, val in param_state_dict.items():
        print(f"name:{name}, params:{val}")

if __name__ == "__main__":
    main()