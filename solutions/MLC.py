from models.BIML_S import BIML_S
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import ARC_gym.utils.metrics as evalUtils

class MLC:

    def __init__(self, input_vocab_size, output_vocab_size, n_enc_layers=3, n_dec_layers=3, emb_size=128,
                 dim_feedforward=512, lr=0.0002, dropout_p=0., k=5, model_path='MLC.pt', load_model=False,
                 device='cuda'):
        self.device = device
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.emb_size = emb_size
        self.dropout_p = dropout_p
        self.SOS_token = torch.tensor([10]).to(device)
        self.output_vocab = np.arange(10)
        self.dim_feedforward = dim_feedforward
        self.act_f = 'gelu'
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.lr = lr
        self.k = k
        self.lr_end_factor = 0.00003

        # If using LR warmup
        self.lr_warmup = False

        # Resume training
        self.load_model = load_model
        self.model_path = model_path

    def evaluate_ll(self, val_dataloader, net, loss_fn=[], p_lapse=0.0):
        # Evaluate the total (sum) log-likelihood across the entire validation set
        #
        # Input
        #   val_dataloader :
        #   net : BIML-S model
        #   langs : dict of dat.Lang classes
        #   p_lapse : (default 0.) combine decoder outputs (prob 1-p_lapse) as mixture with uniform distribution (prob p_lapse)
        net.eval()
        total_N = 0
        total_ll = 0
        if not loss_fn: loss_fn = torch.nn.CrossEntropyLoss()
        val_accuracy = 0.
        for val_batch in val_dataloader:
            val_batch = self.transfer_to_device(val_batch, self.device)

            dict_loss, tmp_accuracy = self.batch_ll(val_batch, net, loss_fn, p_lapse=p_lapse)
            val_accuracy += tmp_accuracy
            total_ll += dict_loss['ll']
            total_N += dict_loss['N']

        val_accuracy /= len(val_dataloader)
        return total_ll, total_N, val_accuracy

    def smooth_decoder_outputs(self, logits_flat, p_lapse, lapse_symb_include):
        # Mix decoder outputs (logits_flat) with uniform distribution over allowed emissions (in lapse_symb_include)
        #
        # Input
        #  logits_flat : (batch*max_len, output_size) # unnomralized log-probabilities
        #  p_lapse : probability of a uniform lapse
        #  lapse_symb_include : list of tokens (strings) that we want to include in the lapse model
        #
        # Output
        #  log_probs_flat : (batch*max_len, output_size) normalized log-probabilities
        lapse_idx_include = [s for s in lapse_symb_include]
        assert self.SOS_token not in lapse_symb_include  # SOS should not be an allowed output through lapse model
        sz = logits_flat.size()  # get size (batch*max_len, output_size)
        probs_flat = F.softmax(logits_flat, dim=1)  # (batch*max_len, output_size)
        num_classes_lapse = len(lapse_idx_include)
        probs_lapse = torch.zeros(sz, dtype=torch.float)
        probs_lapse = probs_lapse.to(device=self.device)
        probs_lapse[:, lapse_idx_include] = 1. / float(num_classes_lapse)
        log_probs_flat = torch.log((1 - p_lapse) * probs_flat + p_lapse * probs_lapse)  # (batch*max_len, output_size)
        return log_probs_flat

    # Target_batches shape is [b x 1 x 5 x 5], because there is only 1 query example
    # required output: [b x 5*5+1] (+1 for the SOS token)
    def shift_flatten_target(self, target_batches):
        def flatten_and_shift(tb):
            # flatten
            tmp = torch.reshape(tb, [-1])

            # shift
            return torch.cat((self.SOS_token, tmp), dim=0)

        shifted_outputs = []
        for tb in target_batches:
            tmp = flatten_and_shift(tb)
            shifted_outputs.append(tmp)

        return torch.stack(shifted_outputs, dim=0).long()

    def batch_ll(self, batch, net, loss_fn, p_lapse=0.0):
        # Evaluate log-likelihood (average over cells, and sum total) for a given batch
        #
        # Input
        #   batch : from dat.make_biml_batch
        #   loss_fn : loss function
        #   langs : dict of dat.Lang classes
        net.eval()
    
        target_batches = batch['yq_padded'].long()  # batch_size x 1 x grid_dim x grid_dim
        target_shift = self.shift_flatten_target(target_batches)  # batch_size x (num_cells + 1)

        # Shifted targets with padding (added SOS symbol at beginning and removed EOS symbol)
        decoder_output = net(target_shift, batch)

        # b*nq x max_length x output_size
        logits_flat = decoder_output[:, :125, :].reshape(-1, decoder_output.shape[-1])  # (batch*max_len, output_size)
        if p_lapse > 0:
            logits_flat = self.smooth_decoder_outputs(logits_flat, p_lapse, self.output_vocab)

        # calculate accuracy
        val_preds = np.argmax(decoder_output[:, :125, :].cpu().data.numpy(), axis=-1)
        val_targets = target_batches.cpu().data.numpy()

        acc = evalUtils.calculateAccuracy(val_preds, val_targets)

        target_batches = torch.reshape(target_batches, [-1])
        loss = loss_fn(logits_flat, target_batches)
        loglike = -loss.cpu().item()
        dict_loss = {}
        dict_loss['ll_by_cell'] = loglike  # average over cells
        dict_loss['N'] = float(target_shift.shape[0] * (target_shift.shape[1]-1))  # total number of valid cells
        dict_loss['ll'] = dict_loss['ll_by_cell'] * dict_loss['N']  # total LL
        return dict_loss, acc

    def train_iter(self, batch, net, loss_fn, optimizer, take_step=True):
        # Update the model for one batch (which is a set of episodes)
        #
        # Input
        #   batch : output from dat.make_biml_batch
        #   net : BIML model
        #   loss_fn : loss function
        #   optimizer : torch optimizer (AdamW)
        #   langs : input and output language class
        #   take_step : if True, update weights. if False, just accumulate gradient
        net.train()
 
        target_batches = batch['yq_padded'].long()  # b*nq x max_length

        target_shift = self.shift_flatten_target(target_batches)  # b*nq x max_length

        # shifted targets with padding (added SOS symbol at beginning and removed EOS symbol)
        decoder_output = net(target_shift, batch)  # b*nq x max_length x output_size
        logits_flat = decoder_output[:, :125, :].reshape(-1, decoder_output.shape[-1])  # (b*nq*max_length, output_size)

        target_batches = torch.reshape(target_batches, [-1])

        loss = loss_fn(logits_flat, target_batches)
        assert (not torch.isinf(loss))
        assert (not torch.isnan(loss))
        loss.backward()
        if take_step:
            optimizer.step()
            optimizer.zero_grad()
        dict_loss = {}
        dict_loss['total'] = loss.cpu().item()

        return dict_loss

    def save_model(self, net):
        state = {'nets_state_dict': net.state_dict()}
        torch.save(state, self.model_path)

    def transfer_to_device(self, batch, device):
        batch['xq+xs+ys_padded'] = batch['xq+xs+ys_padded'].to(device)
        batch['yq_padded'] = batch['yq_padded'].to(device)
        return batch

    def train(self, train_dataloader, val_dataloader, nepochs=50000):

        # setup model
        net = BIML_S(self.emb_size,
                     self.input_vocab_size,
                     self.output_vocab_size,
                     nlayers_encoder=self.n_enc_layers,
                     nlayers_decoder=self.n_dec_layers,
                     dropout_p=self.dropout_p,
                     activation=self.act_f,
                     dim_feedforward=self.dim_feedforward,
                     device=self.device).to(self.device)

        # setup loss and scheduled
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.AdamW(net.parameters(), lr=self.lr, betas=(0.9, 0.95), weight_decay=0.01)

        scheduler_epoch = optim.lr_scheduler.LinearLR(optimizer,
                                                      start_factor=1.0,
                                                      end_factor=self.lr_end_factor,
                                                      total_iters=nepochs - 1,
                                                      verbose=False)

        avg_train_loss = 0.
        best_val_loss = float('inf')
        counter = 0  # num updates since the loss was last reported
        step = 0
        epoch_start = 1

        import matplotlib.pyplot as plt

        if self.load_model:
            print("Loading model from ", self.model_path)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            net.load_state_dict(checkpoint['nets_state_dict'])

        train_losses = []
        val_losses = []
        val_accuracies = []
        print('Setting LR={:.7f}'.format(optimizer.param_groups[0]['lr']))

        for epoch in range(epoch_start, nepochs + 1):
            print("Epoch #%i" % epoch)
            epoch_train_loss = 0.
            epoch_val_loss = 0.
            epoch_accuracy = 0.
            num_batches = 0
            for batch_idx, train_batch in enumerate(train_dataloader):
                train_batch = self.transfer_to_device(train_batch, self.device)

                dict_loss = self.train_iter(train_batch, net, loss_fn, optimizer, take_step=True)
                step += 1
                avg_train_loss += dict_loss['total']
                counter += 1

                with torch.no_grad():
                    # compute val loss
                    total_ll, total_N, val_accuracy = self.evaluate_ll(val_dataloader, net, loss_fn=loss_fn)
                    val_loss = -total_ll / total_N
                    val_losses.append(val_loss)
                    epoch_train_loss += dict_loss['total']
                    epoch_val_loss += val_loss
                    epoch_accuracy += val_accuracy
                    num_batches += 1
                    val_accuracies.append(val_accuracy)
                    train_losses.append(dict_loss['total'])

                    avg_train_loss = 0.
                    counter = 0

                if val_loss < best_val_loss and (epoch > 25):  # don't bother saving best model in early epochs
                    best_val_loss = val_loss
                    self.save_model(net)

            epoch_train_loss /= num_batches
            epoch_val_loss /= num_batches
            epoch_accuracy /= num_batches
            print('TrainLoss: %.4f, ValLoss: %.4f, ValAccuracy: %.2f' % (epoch_train_loss,
                                                                         epoch_val_loss,
                                                                         epoch_accuracy * 100.))
            # after each epoch, adjust the general learning rate
            scheduler_epoch.step()

        print('Training complete.')

        plt.plot(val_losses, color='orange')
        plt.plot(train_losses, color='blue')
        plt.show()