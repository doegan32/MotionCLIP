import torch
from tqdm import tqdm
from copy import deepcopy
import pickle
import clip


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def train_or_test(model, optimizer, iterator, device, mode="train"):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    print("train_or_test")
    print(mode)
    print(grad_env)

    # loss of the epoch
    dict_loss = {}
    with grad_env():
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):



            # Put everything in device
            # Added if is_tensor as 'clip_text' in batch is a list of strings, not a tensor!
            batch = {key: val.to(device) if torch.is_tensor(val) else val for key, val in batch.items()}

            print("in training")
            print(type(batch))
            for k in batch.keys():
                print(k, type(batch[k]))

            print("ending batch")
            print("x", batch['x'].shape)
            print("y", batch['y'].shape)
            print("mask", batch['mask'].shape)
            print("lengths", batch['lengths'].shape)
            print("clip_images", batch['clip_images'].shape)
            print("clip_text", len(batch['clip_text']))
            print("clip_path", len(batch['clip_path']))


            if mode == "train":
                # update the gradients to zero
                optimizer.zero_grad()

            # forward pass
            batch = model(batch)

            mixed_loss, losses = model.compute_loss(batch)

            if i == 0:
                dict_loss = deepcopy(losses)
            else:
                for key in dict_loss.keys():
                    dict_loss[key] += losses[key]

            if mode == "train":
                # backward pass
                mixed_loss.backward()
                # update the weights
                if model.clip_training:
                    convert_models_to_fp32(model.clip_model)
                    optimizer.step()
                    clip.model.convert_weights(model.clip_model)
                else:
                    print("we aint clip training?")
                    optimizer.step()

    return dict_loss


def train(model, optimizer, iterator, device):
    print("in training")
    return train_or_test(model, optimizer, iterator, device, mode="train")


def test(model, optimizer, iterator, device):
    print("in testing")
    return train_or_test(model, optimizer, iterator, device, mode="test")
