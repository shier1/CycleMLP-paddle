import matplotlib.pyplot as plt

epochs = []
lrs = []

with open("./lr_info.txt", "r") as f:
    for line in f.readlines():
        epoch = eval(line.split(' ')[5].rstrip('.'))
        lr = eval(line.split(' ')[6].split('=')[1])
        epochs.append(epoch)
        lrs.append(lr)
lr_1 = 1e-5
lr_2 = 1e-4
lr_3 = 1e-6
lr_4 = 5e-5

ax = plt.subplot()
ax.plot(epochs, lrs,c='g')
ax.axhline(lr_1, c='r')
ax.axhline(lr_2, c='r')
ax.axhline(lr_3, c='r')
ax.axhline(lr_4, c='r')
plt.show()
