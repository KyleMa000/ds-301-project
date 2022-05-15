import matplotlib.pyplot as plt

loss_file = open('logs4/mobilenetv3/log.txt', 'r')

train_loss = []
test_loss = []
for line in loss_file:
    loss_arr = line.split(',')
    train_loss.append(float(loss_arr[0]))
    test_loss.append(float(loss_arr[1]))

x = [ii for ii in range(100)]
plt.plot(x, train_loss)
plt.plot(x, test_loss)
plt.show()