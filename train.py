# @Author: charles
# @Date:   2020-04-21T10:28:17-04:00
# @Last modified by:   charles
# @Last modified time: 2020-04-21T15:56:57-04:00


from make_data import syn_dataloader, values, Nw, Ns
from autoencoder import BalancedAE
from autoencoder import TiedAutoEncoderFunctional
from autoencoder import TiedAutoEncoder
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


model = TiedAutoEncoderFunctional(Nw, 32)
optimizer = optim.Adam(model.parameters(), lr=0.5)
criterion = nn.MSELoss()


n_epochs = 100
features = np.empty((n_epochs, len(syn_dataloader), 32))
losses = []
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(syn_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        feats, outputs = model(inputs)

        features[epoch, i] = feats.detach().numpy()
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 1000 == 0:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

    # print epoch loss
    epoch_loss = running_loss / len(syn_dataloader)
    losses.append(epoch_loss)
    print([epoch+1, epoch_loss])

print('Finished Training')

features = features.mean(axis=0)

distances = np.empty((len(features)))
for i in range(len(features)):
    num = np.linalg.norm(features[i] - features[i-1])
    den = np.sqrt(np.linalg.norm(features[i]) * np.linalg.norm(features[i-1]))
    distances[i] = num/den


peaks, _ = find_peaks(distances, prominence=(0.9, None))
plt.figure()
plt.plot(distances)
plt.plot(peaks, distances[peaks], 'o', color='C3')
plt.xlabel('Window number')
plt.ylabel('Feature dist')

plt.figure()
plt.plot(values)
plt.fill_between(range(len(values)), values, alpha=0.5)
[plt.axvline(Ns*_x, linewidth=1, color='C3') for _x in peaks]
plt.ylim([0, None])
plt.xlabel('Time')
plt.ylabel('Signal')
