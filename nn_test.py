import NeuralNetwork
import MNISTdata
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    train_data, validation_data, test_data = MNISTdata.load_data()

    img_dim = 28
    
    nn = NeuralNetwork.FullyConnectedNetwork([img_dim * img_dim, 30, 10])
    nn.train_SGD(train_data, num_epochs=30, batch_size=10, learning_rate=3.0,
                 test_data=validation_data)
    print("Final Test: {0}/{1}".format(nn.evaluate(test_data), len(test_data[1])))

    num_plots = 20
    offset = 1234
    titles = np.argmax(nn.feedforward(test_data[0][offset:offset+num_plots]), axis = 0)
    rows = int(num_plots**0.5)
    cols = math.ceil(num_plots / rows)
    fig, axs = plt.subplots(rows, cols)
    for ifig in range(num_plots):
        axs[int(ifig/cols), ifig%cols].imshow(np.reshape(test_data[0][offset + ifig], (img_dim, img_dim)))
        axs[int(ifig/cols), ifig%cols].set_title(titles[ifig])
        axs[int(ifig/cols), ifig%cols].axis('off')
    plt.savefig('out.png')
    plt.show()
