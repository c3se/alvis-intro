import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import horvod.torch as hvd

from model import Model
from dataset import RandomDataset


def setup(verbose=False):

    hvd.init()

    if verbose:
        print(f'''
=============================================
Rank: {hvd.rank()}
Local rank: {hvd.local_rank()}
World size: {hvd.size()}
=============================================
        ''')

    # Limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)
    

def cleanup():
    pass


def run_process():
    '''Run process

    This is what is actually run on each process.
    '''
    # Setup this process
    setup(verbose=True)
    
    # Initialize data_loader
    input_size = 5
    output_size = 1
    batch_size = 30
    data_size = 100

    data_loader = DataLoader(
        dataset=RandomDataset(input_size, data_size),
        batch_size=batch_size,
        shuffle=True,
    )

    # Initialize model and attach to optimizer
    model = Model(input_size, output_size, verbose=False)

    device = torch.device(f"cuda:{hvd.local_rank()}")
    model.to(device)

    opt = optim.SGD(model.parameters(), lr=0.01)

    # Parallelize
    # Broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(opt, root_rank=0)

    # Wrap optimizer with DistributedOptimizer.
    opt= hvd.DistributedOptimizer(
        opt,
        named_parameters=model.named_parameters(),
    )

    # Actual training
    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        for data, target in data_loader:
            opt.zero_grad()

            input = data.to(device)
            target = target.to(device)
            output = model(input)

            loss = (output - target).pow(2).mean(0)
            loss.backward()
            opt.step()
        
        if hvd.rank()==0:
            print(epoch)

    # Cleanup process
    cleanup()

    return model


def main():
    # Spawn processes
    run_process()


if __name__=="__main__":
    main()