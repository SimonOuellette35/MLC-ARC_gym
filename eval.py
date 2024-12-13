from ARC_gym.MetaDGP import MetaDGP
from solutions.MLC import MLC
from ARC_gym.utils.batching import make_mlc_batch
from torch.utils.data import DataLoader
import ARC_gym.utils.metrics as metrics
import ARC_gym.utils.visualization as viz
import numpy as np

QUANTIFY_OOD = True

NUM_EPOCHS = 25000
train_batch_size = 100
test_batch_size = 10
NUM_TRAIN_TASKS = 1000
NUM_TEST_TASKS = 100

# When both = [3, 5], ValAccuracy ends up at 100%
comp_graph_dist = {
    'train': {
        'num_nodes': [3, 4]
    },
    'test': {
        'num_nodes': [6, 7]
    }
}

grid_dist = {
    'train': {
        'num_pixels': [1, 5],
        'space_dist_x': np.ones(5) / 5.,
        'space_dist_y': np.ones(5) / 5.
    },
    'test': {
        'num_pixels': [1, 5],
        'space_dist_x': np.ones(5) / 5.,
        'space_dist_y': np.ones(5) / 5.
    }
}

# generate batches of data and visualize them to see if it makes sense.
dgp = MetaDGP()
meta_train_dataset, meta_test_dataset, meta_train_tasks, meta_test_tasks = dgp.instantiateExperiment(
    trainN=NUM_TRAIN_TASKS,
    testN=NUM_TEST_TASKS,
    num_modules=12,
    comp_graph_dist=comp_graph_dist,
    grid_dist=grid_dist,
    max_graphs=2000,
    augment_data=False)

meta_train_dataloader = DataLoader( meta_train_dataset,
                                    batch_size=train_batch_size,
                                    collate_fn=lambda x:make_mlc_batch(x),
                                    shuffle=True)
meta_test_dataloader = DataLoader(  meta_test_dataset,
                                    batch_size=test_batch_size,
                                    collate_fn=lambda x:make_mlc_batch(x),
                                    shuffle=False)

if QUANTIFY_OOD:
        OOD = metrics.quantify_comp_graph_OOD(meta_train_dataset, meta_train_tasks,
                                              meta_test_dataset, meta_test_tasks, dgp.modules)

        print("OOD = ", OOD)

# use the MLC method to solve the meta-test set. What kind of accuracy are we seeing?
model = MLC(input_vocab_size=13, # + 3 for SOS, ITEM, I/O separator tokens
            output_vocab_size=13,
            k=5,
            device='cuda')

model.train(meta_train_dataloader, meta_test_dataloader, nepochs=NUM_EPOCHS)
