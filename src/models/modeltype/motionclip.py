import numpy as np
import torch
import torch.nn as nn

import clip
from ..tools.losses import get_loss_function
from ..rotation2xyz import Rotation2xyz

loss_ce = nn.CrossEntropyLoss()
loss_mse = nn.MSELoss()

cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
from tqdm import tqdm


class MOTIONCLIP(nn.Module):
    def __init__(self, encoder, decoder, device, lambdas, latent_dim, outputxyz,
                 pose_rep, glob, glob_rot, translation, jointstype, vertstrans, clip_lambdas={}, **kwargs):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.outputxyz = outputxyz # True for viz_clip_text anwways
        print("self.outputxyz: ", self.outputxyz)

        self.lambdas = lambdas
        self.clip_lambdas = clip_lambdas

        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.device = device
        self.translation = translation
        self.jointstype = jointstype
        self.vertstrans = vertstrans

        self.clip_model = kwargs['clip_model']
        self.clip_training = kwargs.get('clip_training', False)
        if self.clip_training and self.clip_model:
            self.clip_model.training = True
        else:
            if self.clip_model:
                assert self.clip_model.training == False  # make sure clip is frozen

        self.losses = list(self.lambdas) + ["mixed"]

        self.rotation2xyz = Rotation2xyz(device=self.device)
        self.param2xyz = {"pose_rep": self.pose_rep, # rot6d
                          "glob_rot": self.glob_rot, #  [3.141592653589793, 0, 0]
                          "glob": self.glob, # True
                          "jointstype": self.jointstype, # smpl
                          "translation": self.translation, # True
                          "vertstrans": self.vertstrans} # False
        
        print("param2xyz")
        for key in self.param2xyz.keys():
            print(key,self.param2xyz[key] )

    def rot2xyz(self, x, mask, get_rotations_back=False, **kwargs):
        kargs = self.param2xyz.copy()
        kargs.update(kwargs)
        return self.rotation2xyz(x, mask, get_rotations_back=get_rotations_back, **kargs)

    def compute_loss(self, batch):

        # compute all losses other than clip
        mixed_loss = 0.
        losses = {}
        print("computer losses")
        for ltype, lam in self.lambdas.items():
            print(ltype)
            loss_function = get_loss_function(ltype)
            loss = loss_function(self, batch)
            mixed_loss += loss * lam
            losses[ltype] = loss.item()

        # compute clip losses
        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)

        # mix and add clip losses
        mixed_loss_with_clip = mixed_loss + mixed_clip_loss  # this is the ultimate loss to optimize, combining ALL losses
        losses.update(clip_losses)
        losses["mixed_without_clip"] = mixed_loss.item()
        losses["mixed_clip_only"] = mixed_clip_loss if isinstance(mixed_clip_loss, float) else mixed_clip_loss.item()
        losses["mixed_with_clip"] = mixed_loss_with_clip if isinstance(mixed_loss_with_clip,
                                                                       float) else mixed_loss_with_clip.item()

        return mixed_loss_with_clip, losses

    def compute_clip_losses(self, batch):
        mixed_clip_loss = 0.
        clip_losses = {}

        if self.clip_training:
            for d in self.clip_training.split('_'):
                if d == 'image':
                    features = self.clip_model.encode_image(
                        batch['clip_images']).float()  # preprocess is done in dataloader
                elif d == 'text':
                    texts = clip.tokenize(batch['clip_text']).to(self.device)
                    features = self.clip_model.encode_text(texts).float()

                # normalized features
                features_norm = features / features.norm(dim=-1, keepdim=True)
                seq_motion_features_norm = batch["z"] / batch["z"].norm(dim=-1, keepdim=True)
                logit_scale = self.clip_model.logit_scale.exp()
                logits_per_motion = logit_scale * seq_motion_features_norm @ features_norm.t()
                logits_per_d = logits_per_motion.t()

                batch_size = batch['x'].shape[0]
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=self.device)

                ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
                ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
                clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

                clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss.item()
                clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss.item()
                clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss.item()
                mixed_clip_loss += clip_mixed_loss
        else:
            for d in self.clip_lambdas.keys():
                if len(self.clip_lambdas[d].keys()) == 0:
                    continue
                with torch.no_grad():
                    if d == 'image':
                        features = self.clip_model.encode_image(
                            batch['clip_images']).float()  # preprocess is done in dataloader
                    elif d == 'text':
                        texts = clip.tokenize(batch['clip_text']).to(self.device)
                        features = self.clip_model.encode_text(texts).float()
                    else:
                        raise ValueError(f'Invalid clip domain [{d}]')

                # normalized features
                features_norm = features / features.norm(dim=-1, keepdim=True)
                seq_motion_features_norm = batch["z"] / batch["z"].norm(dim=-1, keepdim=True)

                if 'ce' in self.clip_lambdas[d].keys():
                    logit_scale = self.clip_model.logit_scale.exp()
                    logits_per_motion = logit_scale * seq_motion_features_norm @ features_norm.t()
                    logits_per_d = logits_per_motion.t()

                    batch_size = batch['x'].shape[0]
                    ground_truth = torch.arange(batch_size, dtype=torch.long, device=self.device)

                    ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
                    ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
                    clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

                    clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss.item()
                    clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss.item()
                    clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss.item()
                    mixed_clip_loss += clip_mixed_loss * self.clip_lambdas[d]['ce']

                if 'mse' in self.clip_lambdas[d].keys():
                    mse_clip_loss = loss_mse(features, batch["z"])
                    clip_losses[f'{d}_mse'] = mse_clip_loss.item()
                    mixed_clip_loss += mse_clip_loss * self.clip_lambdas[d]['mse']

                if 'cosine' in self.clip_lambdas[d].keys():
                    cos = cosine_sim(features_norm, seq_motion_features_norm)
                    cosine_loss = (1 - cos).mean()
                    clip_losses[f'{d}_cosine'] = cosine_loss.item()
                    mixed_clip_loss += cosine_loss * self.clip_lambdas[d]['cosine']

        return mixed_clip_loss, clip_losses

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def generate_one(self, cls, duration, fact=1, xyz=False):
        y = torch.tensor([cls], dtype=int, device=self.device)[None]
        lengths = torch.tensor([duration], dtype=int, device=self.device)
        mask = self.lengths_to_mask(lengths)
        z = torch.randn(self.latent_dim, device=self.device)[None]

        batch = {"z": fact * z, "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)

        if not xyz:
            return batch["output"][0]

        output_xyz = self.rot2xyz(batch["output"], batch["mask"])

        return output_xyz[0]

    def generate(self, classes, durations, nspa=1,
                 # noise_same_action="random", noise_diff_action="random",
                 # fact=1,
                 is_amass=False, is_clip_features=False,
                 # input_type="motion",
                 textual_labels=None):
        clip_dim = self.clip_model.ln_final.normalized_shape[0]
        if is_clip_features:
            # assumed dims: classes [nspa, nats, 512]
            assert len(classes.shape) == 3
            assert classes.shape[-1] == clip_dim
            clip_features = classes.reshape([-1, clip_dim])
            nspa, nats = classes.shape[:2]
            # y = torch.zeros(y_action_names.shape, dtype=int)
            y = clip_features
            print("clip_features: ", clip_features.shape) # 24 x 512
            print(nspa) # 1
            print(nats) # 24

            if textual_labels is not None:
                print("textual labels are not None")
                y = np.array(textual_labels).reshape([-1])

            

        if len(durations.shape) == 1:
            lengths = durations.to(self.device).repeat(nspa)
            print("lenghts 1: ", lengths.shape)
        else:
            lengths = durations.to(self.device).reshape(clip_features.shape[0])  # 24
            print("lenghts 2: ", lengths.shape)

        mask = self.lengths_to_mask(lengths)  # 24 x 120, this is all trues - so evidently not to block out viewing future frames
        print("mask: ", mask.shape)
        print(torch.sum(mask * 1.0))

        batch = {"z": clip_features,  #  24 x 512
                 
                 "y": y,                # 24 x 12 just clip features as well, i.e. z and y are the same (at least for text2motion)
                 "mask": mask,  #  24 x 120
                 "lengths": lengths}  # 24
        


        if not is_clip_features:
            batch['y'] = y

        print("batch")
        print(torch.sum(batch['z']))
        print(torch.sum(batch['y']))


        batch = self.decoder(batch)
        print("back to generator for output: ")
        for key in batch.keys():
            print(key, batch[key].shape)

        # keys are z,y,mask,lengths, output where output shape is 24 x 25 x 6 x 120 (batch, jpints, features, frames)
     

        if is_amass:  # lose global orientation for amass dataset
            batch['output'][:, 0] = torch.tensor([1, 0, 0, 0, -1, 0]).unsqueeze(0).unsqueeze(2)
            print("batch['output'][:, 0] ", batch['output'][:, 0].shape)

        if self.outputxyz: # true
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
            print(" batch[output_xyz] ",  batch["output_xyz"].shape) # 24 x 24 x 3 x 120 - so we lose one joint here
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]

        return  batch

    def generate_from_embedding(self, classes, durations, nspa=1, is_amass=False, classes_gaussians=None):

        if nspa is None:
            nspa = 1
        nats = len(classes)

        y = classes.to(self.device).repeat(nspa)  # (view(nspa, nats))
        if len(durations.shape) == 1:
            lengths = durations.to(self.device).repeat(nspa)
        else:
            lengths = durations.to(self.device).reshape(y.shape)
        mask = self.lengths_to_mask(lengths)
        classes_np = classes.cpu().detach().numpy()

        motion_samples_ = np.zeros((classes_np.shape[0], 512), dtype='float32')
        for class_label in tqdm(np.unique(classes_np), total=len(np.unique(classes_np))):
            class_mask = np.where(classes_np == class_label)[0]
            sample_mu = classes_gaussians[class_label]['mu']
            sample_var = classes_gaussians[class_label]['var']

            sample = np.random.multivariate_normal(sample_mu, sample_var, size=len(class_mask))
            motion_samples_[class_mask, :] = sample

        zz = torch.from_numpy(motion_samples_).to(self.device)

        batch = {"z": zz,
                 "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)

        if is_amass:  # lose global orientation for amass dataset
            batch['output'][:, 0] = torch.tensor([1, 0, 0, 0, -1, 0]).unsqueeze(0).unsqueeze(2)

        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        return batch

    def forward(self, batch):
        if self.outputxyz:
            batch["x_xyz"] = self.rot2xyz(batch["x"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]
        # encode
        batch.update(self.encoder(batch))

        print("output of forward pass encoder")
        print(" batch[mu] ",  batch["mu"].shape)

        batch["z"] = batch["mu"]
        # decode
        batch.update(self.decoder(batch))

        # if we want to output xyz
        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        return batch
