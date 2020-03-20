import model_chen_2019 as mc19
import oxfordDataLoader as odl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Note: Why do I need to re-import torch?
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# The main area
if __name__=="__main__":

    # -------------
    # Get the data:
    # batch_size = 16
    batch_size = 256 # from paper
    z = odl.OxfordDataset()
    loader = DataLoader(z, batch_size=batch_size, pin_memory=True)

    # ----------------
    # Setup the model:
    n_epochs = 10
    #lr = 0.01
    lr = 0.00001 # paper has 1e-5
    criterion = nn.MSELoss()

    input_size  = 6
    hidden_size = 128 # 256
    num_layers  = 3   # 2
    output_size = 3   # 4

    model = mc19.Chen_IEEE(input_size, hidden_size, num_layers, output_size)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hidden    = None


    # Now train:
    for epoch in range(1, n_epochs+1):
        # Begin your epoch by zeroing your gradients in
        # your optimizer (why?)
        optimizer.zero_grad()

        for batch_idx, sample in enumerate(loader):

            # Concatenate inputs attitude and acceleration
            # from the sensor along the 2nd dimension,
            # which is basically the index for the data group
            data_in = torch.cat((sample[1], sample[4]), 2).to(device)

            # A note on permuting:
            # if you have a matrix, the zeroth 'axis' is the
            # column, and the first is the row. So,
            # permute(1,0) will put the columns into rows,
            # and the rows into columns. It expands to nth axis
            # with tensors.
            # For this particular operation, we are swapping the
            # zeroth and first axes, and those are:
            # og_zeroth = index
            # og_first  = physical values (I think)
            data_in.permute(1,0,2)
            trans = sample[7].to(device) # translation from truth


            # Get the diff. TODO: Understand this better!
            delta_trans = torch.cat((torch.zeros(trans.shape[0], 1, 3, dtype=torch.float32).to(device), (trans[:,1:,:] - trans[:,:-1,:])), 1)
            output, hidden = model(data_in, hidden)


            # Note, output and delta_trans must have same shape (?)
            loss = criterion(output, delta_trans)
            if batch_idx != len(loader):
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            optimizer.step()

        # Small number of epochs, so print all:
        print('Epoch: {}/{}..........'.format(epoch, n_epochs), end='')
        print('Loss: {:0.4f}'.format(loss.item()))




