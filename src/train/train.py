import os
import sys
sys.path.append('.')

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from src.train.trainer import train
from src.utils.tensors import collate
import src.utils.fixseed  # noqa

from src.parser.training import parser
from src.utils.get_model_and_data import get_model_and_data


def do_epochs(model, datasets, parameters, optimizer, writer):

    print("in do_epochs")

    dataset = datasets["train"]
    train_iterator = DataLoader(dataset, batch_size=parameters["batch_size"],
                                shuffle=True, collate_fn=collate) #, num_workers=8, collate_fn=collate)

    logpath = os.path.join(parameters["folder"], "training.log")
    with open(logpath, "w") as logfile:
        for epoch in range(1, parameters["num_epochs"]+1):

            print("in an epoch ")

            dict_loss = train(model, optimizer, train_iterator, model.device)

            for key in dict_loss.keys():
                dict_loss[key] /= len(train_iterator)
                writer.add_scalar(f"Loss/{key}", dict_loss[key], epoch)

            epochlog = f"Epoch {epoch}, train losses: {dict_loss}"
            print(epochlog)
            print(epochlog, file=logfile)

            if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["num_epochs"]):
                checkpoint_path = os.path.join(parameters["folder"],
                                               'checkpoint_{:04d}.pth.tar'.format(epoch))
                print('Saving checkpoint {}'.format(checkpoint_path))
                if parameters.get('clip_training', '') == '':
                    state_dict_wo_clip = {k: v for k,v in model.state_dict().items() if not k.startswith('clip_model.')}
                else:
                    state_dict_wo_clip = model.state_dict()
                torch.save(state_dict_wo_clip, checkpoint_path)

            writer.flush()


if __name__ == '__main__':
    # parse options
    parameters = parser()
    # print(parameters.batch_size)

    # logging tensorboard

    print("parameters[folder] ", parameters["folder"])
    parameters["folder"] = "./exps/"
    writer = SummaryWriter(log_dir=parameters["folder"])

    device = parameters["device"] #"cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    # parameters = {}
    # parameters["device"] = "cuda"

    model, datasets = get_model_and_data(parameters)

    # print("length of dataset ", len(datasets["train"]))


    print(type(datasets))
    for k in datasets.keys():
        print(k, type(datasets[k]))
        for i in [1000,2000,5000,100, 10000, 15000, 20000, 25000]: # range(2):
            x = datasets[k].__getitem__(i)
            print(type(x))
            for j in x.keys():
                print(j, type(x[j]))
            print("inp: ", x["inp"].shape)
            print("target ", x["target"])
            print("clip_image ", x["clip_image"].shape)
            print("clip_path ", x["clip_path"])
            print("clip_text ", x["clip_text"])


    # optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"])

    # print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # print("Training model..")

    # do_epochs(model, datasets, parameters, optimizer, writer)

    writer.close()


