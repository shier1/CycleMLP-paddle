from ast import arguments
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser("drawing")
parser.add_argument("-log_file_path", type=str, default='output/Epoch_val_info.txt')
arguments = parser.parse_args()


def main():
    all_acc1 = []
    all_acc5 = []
    epochs = []

    with open(arguments.log_file_path, 'r') as f:
        for line in f.readlines():
            if "after" in line:
                epoch = eval(line.split(' ')[-1])
                epochs.append(epoch)
            else:
                acc1 = eval(line.split(' ')[9].rstrip(','))
                acc5 = eval(line.split(' ')[-3].rstrip(','))
                all_acc1.append(acc1)
                all_acc5.append(acc5)

    ax = plt.subplot()
    ax.plot(epochs, all_acc1, color='tab:pink', label='acc1')
    ax.plot(epochs, all_acc5, c='tab:cyan',label='acc5')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()