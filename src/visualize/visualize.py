import os
import imageio
import sys
sys.path.append('.')

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.visualize.anim import plot_3d_motion_dico, load_anim
import clip
from PIL import Image
import pickle
import src.utils.rotation_conversions as geometry
from textwrap import wrap
import shutil
import subprocess as sp
from copy import deepcopy
 # import cv2

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio


import numpy as np

import matplotlib.animation 
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt


def plot_frame(joints, frame_idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = joints[:, 0, frame_idx]
    y = joints[:, 2, frame_idx]
    z = -joints[:, 1, frame_idx]

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.close(fig)
    return fig



def render_video(joint_sequence, output_file):
    frames = []
    num_frames = joint_sequence.shape[2]

    for frame_idx in range(num_frames):
        fig = plot_frame(joint_sequence, frame_idx)
        fig.canvas.draw()

        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

    imageio.mimsave(output_file, frames, fps=30)

GPU_MINIMUM_MEMORY = 5500

def stack_images(real, real_gens, gen):
    nleft_cols = len(real_gens) + 1
    print("Stacking frames..")
    allframes = np.concatenate((real[:, None, ...], *[x[:, None, ...] for x in real_gens], gen), 1)
    nframes, nspa, nats, h, w, pix = allframes.shape
    blackborder = np.zeros((w//30, h*nats, pix), dtype=allframes.dtype)
    frames = []
    for frame_idx in tqdm(range(nframes)):
        columns = np.vstack(allframes[frame_idx].transpose(1, 2, 3, 4, 0)).transpose(3, 1, 0, 2)
        frame = np.concatenate((*columns[0:nleft_cols], blackborder, *columns[nleft_cols:]), 0).transpose(1, 0, 2)
        frames.append(frame)
    return np.stack(frames)

def stack_gen_and_images(gen, images):
    # nleft_cols = len(real_gens) + 1
    print("Stacking frames..")
    allframes = np.concatenate((images, gen), 2)
    nframes, nspa, nats, h, w, pix = allframes.shape
    blackborder = np.zeros((w//30, h*nats, pix), dtype=allframes.dtype)
    frames = []
    for frame_idx in tqdm(range(nframes)):
        columns = np.vstack(allframes[frame_idx].transpose(1, 2, 3, 4, 0)).transpose(3, 1, 0, 2)
        frame = np.concatenate((columns[:]), 0).transpose(1, 0, 2)
        frames.append(frame)
    return np.stack(frames)

def stack_gen_only(gen):
    # nleft_cols = len(real_gens) + 1
    print("Stacking frames..")
    # allframes = np.concatenate((real[:, None, ...], *[x[:, None, ...] for x in real_gens], gen), 1)
    allframes = gen
    nframes, nspa, nats, h, w, pix = allframes.shape
    blackborder = np.zeros((w//30, h*nats, pix), dtype=allframes.dtype)
    frames = []
    for frame_idx in tqdm(range(nframes)):
        columns = np.vstack(allframes[frame_idx].transpose(1, 2, 3, 4, 0)).transpose(3, 1, 0, 2)
        frame = np.concatenate((columns[:]), 0).transpose(1, 0, 2)
        frames.append(frame)
    return np.stack(frames)


def generate_by_video(visualization, reconstructions, generation,
                      label_to_action_name, params, nats, nspa, tmp_path, image_pathes=None, mode=None):
    # shape : (17, 3, 4, 480, 640, 3)
    # (nframes, row, column, h, w, 3)
    fps = params["fps"]

    params = params.copy()

    if "output_xyz" in visualization or "output_xyz" in generation:
        outputkey = "output_xyz"
        params["pose_rep"] = "xyz"
    else:
        outputkey = "poses"
    keep = [outputkey, "lengths", "y"]

    def _to_np(x):
        if type(x).__module__ == np.__name__:
            return x
        else:  # assume tensor
            return x.data.cpu().numpy()
    # print(visualization)
    # input(2)
    visu = {key: _to_np(visualization[key]) for key in keep if key in visualization.keys()}
    recons = {mode: {key: _to_np(reconstruction[key]) for key in keep if key in reconstruction.keys()}
              for mode, reconstruction in reconstructions.items()}
    print(generation)
    input(1)
    gener = {key: _to_np(generation[key]) for key in keep if key in generation.keys()}
    print(generation.keys())
    for kk in generation.keys():
        print(kk)
        print(generation[kk].shape)
    input(2)
    def get_palette(i, nspa):
        if mode == 'edit' and i < 3:
            return 'orange'
        elif mode == 'interp' and i in [0, nspa-1]:
            return 'orange'
        return 'blue'


    if(len(visu) > 0):
        lenmax = max(gener["lengths"].max(),
                     visu["lengths"].max())
    else:
        lenmax = gener["lengths"].max()
    timesize = lenmax + 5
    # if params['appearance_mode'] == 'motionclip':
    #     timesize = lenmax + 20

    import multiprocessing

    def pool_job_with_desc(pool, iterator, desc, max_, save_path_format, isij):
        with tqdm(total=max_, desc=desc.format("Render")) as pbar:
            for _ in pool.imap_unordered(plot_3d_motion_dico, iterator):
                pbar.update()
        if isij:
            array = np.stack([[load_anim(save_path_format.format(i, j), timesize)
                               for j in range(nats)]
                              for i in tqdm(range(nspa), desc=desc.format("Load"))])
            return array.transpose(2, 0, 1, 3, 4, 5)
        else:
            array = np.stack([load_anim(save_path_format.format(i), timesize)
                              for i in tqdm(range(nats), desc=desc.format("Load"))])
            return array.transpose(1, 0, 2, 3, 4)

    with multiprocessing.Pool() as pool:
        # Generated samples
        save_path_format = os.path.join(tmp_path, "gen_{}_{}.gif")
        iterator = ((gener[outputkey][i, j],
                     gener["lengths"][i, j],
                     save_path_format.format(i, j),
                     # params, {"title": f"gen: {label_to_action_name(gener['y'][i, j])}", "interval": 1000/fps})
                     params, {"title": f"walk", "interval": 1000/fps, "palette": get_palette(i, nspa)})
                    for j in range(nats) for i in range(nspa))
        gener["frames"] = pool_job_with_desc(pool, iterator,
                                             "{} the generated samples",
                                             nats*nspa,
                                             save_path_format,
                                             True)

        # Make frames with no title blank
        frames_no_title = gener['y'] == ''
        gener["frames"][:, frames_no_title] = gener["frames"][:, 0, 0:1, 0:1, 0:1] # cast the corner pixel value for all blank box

        # Real samples
        if len(visu) > 0:
            save_path_format = os.path.join(tmp_path, "real_{}.gif")
            iterator = ((visu[outputkey][i],
                         visu["lengths"][i],
                         save_path_format.format(i),
                         params, {"title": f"real: {label_to_action_name(visu['y'][i])}", "interval": 1000/fps})
                        for i in range(nats))
            visu["frames"] = pool_job_with_desc(pool, iterator,
                                                "{} the real samples",
                                                nats,
                                                save_path_format,
                                                False)
        for mode, recon in recons.items():
            # Reconstructed samples
            save_path_format = os.path.join(tmp_path, f"reconstructed_{mode}_" + "{}.gif")
            iterator = ((recon[outputkey][i],
                         recon["lengths"][i],
                         save_path_format.format(i),
                         params, {"title": f"recons: {label_to_action_name(recon['y'][i])}",
                                  "interval": 1000/fps})
                        for i in range(nats))
            recon["frames"] = pool_job_with_desc(pool, iterator,
                                                 "{} the reconstructed samples",
                                                 nats,
                                                 save_path_format,
                                                 False)
    if image_pathes is not None:
        # visu["frames"] -> [timesize(65), nspa(n_samples), nats(1), h(290), w(260), n_ch(3)]
        assert nats == 1
        assert nspa == len(image_pathes)
        h, w = gener["frames"].shape[3:5]
        image_frames = []
        for im_path in image_pathes:
            im = Image.open(im_path).resize((w, h))
            image_frames.append(np.tile(np.expand_dims(np.asarray(im)[..., :3], axis=(0, 1, 2)), (timesize, 1, 1, 1, 1, 1)))
        image_frames = np.concatenate(image_frames, axis=1)
        assert image_frames.shape == gener["frames"].shape
        return stack_gen_and_images(gener["frames"], image_frames)

    if len(visu) == 0:
        frames = stack_gen_only(gener["frames"])
    else:
        frames = stack_images(visu["frames"], [recon["frames"] for recon in recons.values()], gener["frames"])
    return frames


def generate_by_video_sequences(visualization, label_to_action_name, params, nats, nspa, tmp_path):
    # shape : (17, 3, 4, 480, 640, 3)
    # (nframes, row, column, h, w, 3)
    fps = params["fps"]

    if "output_xyz" in visualization:
        outputkey = "output_xyz"
        params["pose_rep"] = "xyz"
    else:
        outputkey = "poses"

    keep = [outputkey, "lengths", "y"]
    visu = {key: visualization[key].data.cpu().numpy() for key in keep}
    lenmax = visu["lengths"].max()

    timesize = lenmax + 5
    import multiprocessing

    def pool_job_with_desc(pool, iterator, desc, max_, save_path_format):
        with tqdm(total=max_, desc=desc.format("Render")) as pbar:
            for _ in pool.imap_unordered(plot_3d_motion_dico, iterator):
                pbar.update()
        array = np.stack([[load_anim(save_path_format.format(i, j), timesize)
                           for j in range(nats)]
                          for i in tqdm(range(nspa), desc=desc.format("Load"))])
        return array.transpose(2, 0, 1, 3, 4, 5)

    with multiprocessing.Pool() as pool:
        # Real samples
        save_path_format = os.path.join(tmp_path, "real_{}_{}.gif")
        iterator = ((visu[outputkey][i, j],
                     visu["lengths"][i, j],
                     save_path_format.format(i, j),
                     params, {"title": f"real: {label_to_action_name(visu['y'][i, j])}", "interval": 1000/fps})
                    for j in range(nats) for i in range(nspa))
        visu["frames"] = pool_job_with_desc(pool, iterator,
                                            "{} the real samples",
                                            nats,
                                            save_path_format)
    frames = stack_images_sequence(visu["frames"])
    return frames


def viz_clip_text(model, text_grid, epoch, params, folder):
    """ Generate & viz samples """

    # visualize with joints3D
    model.outputxyz = True

    print(f"Visualization of the epoch {epoch}")

    # noise_same_action = params["noise_same_action"]
    # noise_diff_action = params["noise_diff_action"]

    fact = params["fact_latent"]
    figname = params["figname"].format(epoch)

    classes = np.array(text_grid, dtype=str)
    h, w = classes.shape

    texts = classes.reshape([-1])
    print("texts: ", texts)
    print(texts.shape)
    texts = []
    for i in range(24):
        texts.append("side step")
    texts = np.array(texts, dtype=str)
    clip_tokens = clip.tokenize(texts).to(params['device'])
    print("clip_tokens ", clip_tokens.size())
    clip_features = model.clip_model.encode_text(clip_tokens).float().unsqueeze(0)
    print("clip_features ", clip_features.size())

    gendurations = torch.ones((h*w, 1), dtype=int) * params['num_frames']
    print("nu frames: ", params['num_frames'])
    print(gendurations.size())
    print(gendurations)

    # generate the repr (joints3D/pose etc)
    model.eval()
    with torch.no_grad():

        generation = model.generate(clip_features, gendurations,
                                    is_amass=True,

                                    is_clip_features=True)
        generation['y'] = texts
        print("Generation: ", type(generation))
        print(generation.keys())

    for key, val in generation.items():
        if len(generation[key].shape) == 1:
            generation[key] = val.reshape(h, w)
        else:
            generation[key] = val.reshape(h, w, *val.shape[1:])


    for key, val in generation.items():
        print(key, val.shape) 

    f_name = params['input_file']
    if os.path.isfile(params['input_file']):
        f_name = os.path.basename(params['input_file'].replace('.txt', ''))
    finalpath = os.path.join(folder, 'clip_text_{}_{}'.format(f_name, 'trans_' if params['vertstrans'] else '') + figname + ".gif")
    tmp_path = os.path.join(folder, f"clip_text_subfigures_{figname}")
    print(finalpath)
    print(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)

    # save_pkl(generation['output'], generation['output_xyz'], texts, finalpath.replace('.gif', '.pkl'))
    joint_sequence = generation['output_xyz'][0, 0].reshape(24, 3, params['num_frames'])
    joint_sequence = joint_sequence.cpu().numpy()
    print("joint_sequence ", joint_sequence.shape)
    parents, _ = GetSkeletonInformation("amass")
    positions = np.moveaxis(joint_sequence, 2, 0)
    positions[:,:,1] = - positions[:,:,1]
    PlotAnimation(parents = parents, positions=np.moveaxis(joint_sequence, 2, 0), filename="testing.gif")
    #render_video(joint_sequence, 'output.mp4')

    input(2)

    print("Generate the videos..")
    frames = generate_by_video({}, {}, generation,
                               lambda x: str(x), params, w, h, tmp_path, mode='text')


    print(f"Writing video [{finalpath}]")
    imageio.mimsave(finalpath, frames, fps=params["fps"])


def viz_clip_interp(model, datasets, interp_csv, num_stops, epoch, params, folder):
    """ Generate & viz samples """

    # visualize with joints3D
    model.outputxyz = True

    print(f"Visualization of the epoch {epoch}")
    figname = params["figname"].format(epoch)
    motion_collection = get_motion_text_mapping(datasets)

    # prepare motion representations
    all_clip_features = []
    all_texts = []
    for line in interp_csv:
        # Get CLIP features
        texts = [line['start'], line['end']]
        retrieved_motions = retrieve_motions(datasets, motion_collection, texts, params['device'])
        clip_features = encode_motions(model, retrieved_motions, params['device'])


        # Make interp
        end_factor = np.linspace(0., 1., num=num_stops)
        start_factor = 1. - end_factor
        interp_features = [(start_factor[i]*clip_features[0]) + (end_factor[i]*clip_features[1]) for i in range(num_stops)]
        all_clip_features.append(torch.stack(interp_features))
        texts = texts[:1] + [' '] * (num_stops-2) + texts[-1:]
        all_texts.append(texts)

    all_clip_features = torch.transpose(torch.stack(all_clip_features, axis=0), 0, 1)
    all_texts = np.array(all_texts).T
    h, w = all_clip_features.shape[:2]
    gendurations = torch.ones((h*w, 1), dtype=int) * params['num_frames']

    # generate the repr (joints3D/pose etc)
    model.eval()
    with torch.no_grad():
        generation = model.generate(all_clip_features, gendurations,
                                    is_amass=True,
                                    is_clip_features=True)
        generation['y'] = all_texts.reshape([-1])
        print(type(generation))

    for key, val in generation.items():
        if len(generation[key].shape) == 1:
            generation[key] = val.reshape(h, w)
        else:
            generation[key] = val.reshape(h, w, *val.shape[1:])

    if os.path.isfile(params['input_file']):
        f_name = os.path.basename(params['input_file'].replace('.csv', ''))
    finalpath = os.path.join(folder, f'clip_edit_{f_name}_' + figname + ".gif")
    tmp_path = os.path.join(folder, f"clip_edit_subfigures_{figname}")
    os.makedirs(tmp_path, exist_ok=True)

    print("Generate the videos..")
    frames = generate_by_video({}, {}, generation,
                               lambda x: str(x), params, w, h, tmp_path, mode='interp')

    print(f"Writing video [{finalpath}]")
    imageio.mimsave(finalpath, frames, fps=params["fps"])


def viz_clip_edit(model, datasets, edit_csv, epoch, params, folder):
    """ Generate & viz samples """

    # visualize with joints3D
    model.outputxyz = True

    print(f"Visualization of the epoch {epoch}")
    figname = params["figname"].format(epoch)
    motion_collection = get_motion_text_mapping(datasets)

    # prepare motion representations
    all_clip_features = []
    all_texts = []
    for line in edit_csv:
        # Get CLIP features
        texts = [line['base'], line['v_start'], line['v_end']]
        if line['motion_source'] == 'data':
            retrieved_motions = retrieve_motions(datasets, motion_collection, texts, params['device'])
            clip_features = encode_motions(model, retrieved_motions, params['device'])
        elif line['motion_source'] == 'text':
            clip_tokens = clip.tokenize(texts).to(params['device'])
            clip_features = model.clip_model.encode_text(clip_tokens).float()
        else:
            raise ValueError

        # Make edit
        result_features = clip_features[0] - clip_features[1] + clip_features[2]
        all_clip_features.append(torch.cat([clip_features, result_features.unsqueeze(0)]))
        texts.append('Result')
        all_texts.append(texts)

    all_clip_features = torch.transpose(torch.stack(all_clip_features, axis=0), 0, 1)
    all_texts = np.array(all_texts).T
    h, w = all_clip_features.shape[:2]
    gendurations = torch.ones((h*w, 1), dtype=int) * params['num_frames']

    # generate the repr (joints3D/pose etc)
    model.eval()
    with torch.no_grad():
        generation = model.generate(all_clip_features, gendurations,
                                    is_amass=True,
                                    is_clip_features=True)
        generation['y'] = all_texts.reshape([-1])

    for key, val in generation.items():
        if len(generation[key].shape) == 1:
            generation[key] = val.reshape(h, w)
        else:
            generation[key] = val.reshape(h, w, *val.shape[1:])

    if os.path.isfile(params['input_file']):
        f_name = os.path.basename(params['input_file'].replace('.csv', ''))
    finalpath = os.path.join(folder, f'clip_edit_{f_name}_' + figname + ".gif")
    tmp_path = os.path.join(folder, f"clip_edit_subfigures_{figname}")
    os.makedirs(tmp_path, exist_ok=True)

    print("Generate the videos..")
    frames = generate_by_video({}, {}, generation,
                               lambda x: str(x), params, w, h, tmp_path, mode='edit')

    print(f"Writing video [{finalpath}]")
    imageio.mimsave(finalpath, frames, fps=params["fps"])


def stack_images_sequence(visu):
    print("Stacking frames..")
    allframes = visu
    nframes, nspa, nats, h, w, pix = allframes.shape
    frames = []
    for frame_idx in tqdm(range(nframes)):
        columns = np.vstack(allframes[frame_idx].transpose(1, 2, 3, 4, 0)).transpose(3, 1, 0, 2)
        frame = np.concatenate(columns).transpose(1, 0, 2)
        frames.append(frame)
    return np.stack(frames)


def get_gpu_device():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    for gpu_idx, free_mem in enumerate(memory_free_values):
        if free_mem > GPU_MINIMUM_MEMORY:
            return gpu_idx
    Exception('No GPU with required memory')


def get_motion_text_mapping(datasets):
    print('Building text-motion mapping...')
    split_names = list(datasets.keys())
    collection_path = datasets[split_names[0]].datapath.replace('.pt', '_text_labels.txt')
    if len(split_names) > 1:
        assert split_names[0] in os.path.basename(collection_path)
        _base = os.path.basename(collection_path).replace(split_names[0], 'all')
        collection_path = os.path.join(os.path.dirname(collection_path), _base)
    cache_path = collection_path.replace('.txt', '.npy')

    # load if exists
    word = 'Loading' if os.path.isfile(cache_path) else 'Saving'
    print('{} list of text labels in current dataset to [{}]:'.format(word, collection_path))
    print('Look it up next time you want to retrieve new motions using textual labels.')

    if os.path.isfile(cache_path):
        return np.load(cache_path, allow_pickle=True)[None][0]

    motion_collection = {}
    for split_name, data in datasets.items():
        for i, d in tqdm(enumerate(data)):
            motion_collection[d['clip_text']] = motion_collection.get(d['clip_text'], []) + [(split_name, i)]

    with open(collection_path, 'w') as fw:
        text_labels = sorted(list(motion_collection.keys()))
        fw.write('\n'.join(text_labels) + '\n')
    np.save(cache_path, motion_collection)

    return motion_collection

def retrieve_motions(datasets, motion_collection, texts, device):
    retrieved_motions = []
    for txt in texts:
        _split, _index = motion_collection[txt][0]
        retrieved_motions.append(datasets[_split][_index]['inp'].unsqueeze(0).to(device))
    return torch.cat(retrieved_motions, axis=0)

def encode_motions(model, motions, device):
    return model.encoder({'x': motions,
                          'y': torch.zeros(motions.shape[0], dtype=int, device=device),
                          'mask': model.lengths_to_mask(torch.ones(motions.shape[0], dtype=int, device=device) * 60)})["mu"]







def GetSkeletonInformation(skeletonName, scale=1):

    if skeletonName == "Edinburgh":
        parents = [-1,  0,  1,  2,  3,  4,  2,  6,  7,  8 , 9 , 2 ,11 ,12, 13, 14,  0, 16, 17, 18,  0 ,20 ,21 ,22,  0 ,24, 25]
        offsets = np.array(
            [[  0.  ,        0.       ,   0.       ],
                [  0.     ,     0.        ,  0.       ],
                [ 19.     ,     0.        ,  0.       ],
                [ 22.5    ,     0.6       ,  0.       ],
                [ 14.     ,     0.0308777 ,  0.       ],
                [ 17.     ,     0.        ,  0.       ],
                [ 19.8    ,     3.7       ,  4.3      ],
                [  8.     ,     0.        ,  0.       ],
                [ 15.2    ,     0.        ,  0.       ],
                [ 17.8    ,     0.        ,  0.       ],
                [  7.2    ,     0.        ,  0.       ],
                [ 19.8    ,     3.7       , -4.3      ],
                [  8.     ,     0.        ,  0.151654 ],
                [ 15.2    ,     0.        ,  0.       ],
                [ 17.8    ,     0.        ,  0.       ],
                [  7.2    ,     0.        ,  0.       ],
                [  5.98425,    -7.666     ,  4.78879  ],
                [ 16.     ,     0.        ,  0.       ],
                [ 18.     ,     0.        ,  0.       ],
                [  0.     ,   -10.8       ,  0.       ],
                [  5.98425,    -7.66598   , -4.78879  ],
                [ 16.     ,     0.        ,  0.       ],
                [ 18.     ,     0.        ,  0.       ],
                [  0.     ,   -10.8       ,  0.       ],
                [  6.83696,    -0.722574  ,  0.       ],
                [ 12.     ,     0.        ,  0.       ],
                [ 12.     ,     0.        ,  0.       ]]
        )
    elif skeletonName == "PFNN":
        parents = [-1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 0, 13, 14, 15, 16, 17, 18, 15, 20, 21, 22, 23, 24, 25, 23, 27, 15, 29, 30, 31, 32, 33, 34, 32, 36]
        offsets = np.array(
            [[0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],  
            [1.3631, -1.7946, 0.8393], 
            [2.4481, -6.7261, 0.0000], 
            [2.5622, -7.0396, 0.0000], 
            [0.1576, -0.4331, 2.3226], 
            [0.0000, 0.0000, 0.0000],  
            [0.0000, 0.0000, 0.0000],  
            [-1.3055, -1.7946, 0.8393],
            [-2.5425, -6.9855, 0.0000],
            [-2.5683, -7.0562, 0.0000],
            [-0.1647, -0.4526, 2.3632],
            [0.0000, 0.0000, 0.0000],  
            [0.0000, 0.0000, 0.0000],  
            [0.0283, 2.0356, -0.1934], 
            [0.0567, 2.0488, -0.0428], 
            [0.0000, 0.0000, 0.0000],  
            [-0.0542, 1.7462, 0.1720], 
            [0.1041, 1.7614, -0.1240], 
            [0.0000, 0.0000, 0.0000],  
            [0.0000, 0.0000, 0.0000],  
            [3.3624, 1.2009, -0.3112], 
            [4.9830, -0.0000, -0.0000],
            [3.4836, -0.0000, -0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.7153, -0.0000, -0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [-3.1366, 1.3740, -0.4047],
            [-5.2419, -0.0000, -0.0000],
            [-3.4442, -0.0000, -0.0000],
            [0.0000, 0.0000, 0.0000],
            [-0.6225, -0.0000, -0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000]
                ]
        )
    elif skeletonName == "amass":
        parents = [-1, 0, 0,0, 1, 2, 3, 4, 5, 6, 7, 8, 9,9,9,12,13,14,16,17,18,19,20,21] 
        offsets = 0
    else:
        assert False, 'GetSkeletonInformation, unknown skeletonName'

    return parents, offsets*scale


# helper function to make using colours easier
def ColourNameToNum():
	d = {'red': (255,0,0), 'green': (0,255,0), 'blue':(0,0,255), 'yellow':(255,255,0), 'cyan':(0,255,255), 'magenta':(255,0,255), 'purple':(127,0,127), 'green_dark':(0,127,127), 'yellow_dark':(255, 201, 14)}
	return d

# helper function to enable plotting different body parts in different colours
# All hardcoded for the different skeletons we use, i.e. Bath quadruped, Edinburgh quadruped, Vicon humanoid
def GetDefaultColours(skeletonName):

    colours = ColourNameToNum()
    jointColours = []

    if skeletonName == "Bath":
        jointColours = [
            colours['red'], colours['red'], colours['red'],
            colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'],
            colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'],
            colours['green'], colours['green'], colours['green'], colours['green'], colours['green'],
            colours['green_dark'], colours['green_dark'], colours['green_dark'], colours['green_dark'], colours['green_dark'],
            colours['blue'], colours['blue'], colours['blue'], colours['blue'], colours['blue'],
            colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'],
            colours['yellow_dark'], colours['yellow_dark'], colours['yellow_dark'], colours['yellow_dark'], colours['yellow_dark'], colours['yellow_dark'], colours['yellow_dark'], colours['yellow_dark']
            ]
    elif skeletonName == "Edinburgh":
        jointColours = [
            colours['red'], colours['red'], colours['red'],
            colours['green'], colours['green'],
            colours['green_dark'],
            colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'],
            colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'],
            colours['blue'], colours['blue'], colours['blue'], colours['blue'],
            colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'],
            colours['yellow_dark'], colours['yellow_dark'], colours['yellow_dark']]
    elif skeletonName == "PFNN":
        jointColours = [
            colours['red'],
            colours['blue'], colours['blue'], colours['blue'], colours['blue'], colours['blue'], colours['blue'],
            colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'],
            colours['red'], colours['red'], colours['red'],
            colours['green'], colours['green'],
            colours['green_dark'], colours['green_dark'],
            colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'],
            colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple']
            ]
    elif skeletonName == "Human":
        jointColours = [
            colours['red'], colours['red'], colours['red'], colours['red'], colours['red'],
            colours['green'], colours['green'], colours['green'],
            colours['green_dark'], 
            colours['purple'], colours['purple'], colours['purple'], colours['yellow'], colours['purple'], colours['purple'], colours['purple'], colours['yellow'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], 
            colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], 
            colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'],
            colours['blue'], colours['blue'], colours['blue'], colours['blue'], colours['blue']
        ]
    elif skeletonName == "HumanNoHands":
        jointColours = [colours['red'], colours['red'], colours['red'], colours['red'], colours['red'],
        colours['green'], colours['green'], colours['green'], colours['green_dark'],
        colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'],
        colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'],
        colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'],
        colours['blue'], colours['blue'], colours['blue'], colours['blue'], colours['blue']]
    else:
        assert False, 'GetDefaultColours, unknown skeletonName'

    return jointColours

def PlotFrame(
    animation = None,
    parents = None,
    positions = None,
    frame = None,
    frame_info = None,
    skeleton:str = "Edinburgh", 
    axis_scale=5,
    elev=10,
    azim=180,
    dist=10,
    floorSize=20,
    floorSquares=50,
    display_grid:bool = False,
    ):

    if animation is not None:
        parents = animation.parents
        positions = Animation.positions_global(animation)
    elif (parents is None or positions is None ):
        raise AttributeError("Must provide either animation instance or both positions and parents")

    numJoints = len(parents)

    if animation is not None and frame is None:
        # draw T-pose if no frame number is provided
        positions = Animation.offsets_global(animation)
        title = "T-pose"
    elif frame is None:
        raise AttributeError("Must provide frame number if not providing Animation instance")
    else:
        positions = positions[frame]
        title = "Frame number: {:d}".format(frame)
    
    # deal with provided colour(s)
    if skeleton is not None:
        colours = np.array(GetDefaultColours(skeleton))/255.0
    else: 
        colours = ['red'] * numJoints

    # create figure
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-axis_scale, axis_scale)
    ax.set_zlim3d( -axis_scale, axis_scale)
    ax.set_ylim3d(0, axis_scale)
    ax.set_box_aspect((1,1,0.5), zoom=1)

    if display_grid:
        ax.grid(display_grid)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        ax.set_axis_off()

    # initial camera position
    ax.view_init(elev=elev, azim=azim, vertical_axis='y')
    #ax.dist = dist

    # create floor
    xs = np.linspace(-floorSize, floorSize, floorSquares)
    zs = np.linspace(-floorSize, floorSize, floorSquares)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros(X.shape)
    wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='grey',lw=0.5)

    # display axes
    plt.plot([0,5], [0,0], [0,0], color='r', lw=4)
    plt.plot([0,0], [0,5], [0,0], color='g', lw=4)
    plt.plot([0,0], [0,0], [0,5], color='b', lw=4)
    

    plt.title(title)
    for j in range(1, numJoints): # start at 1 as we don't need line for root
        plt.plot([positions[j,0], positions[parents[j],0]],[positions[j,1], positions[parents[j],1]],[positions[j,2], positions[parents[j],2]], color=colours[j], lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])
    
    try:
        plt.show()
        #plt.save()
    except AttributeError as e:
        pass

    return

def PlotAnimation(
    animation = None,
    parents = None,
    positions = None,
    frame_info = None,
    skeleton:str = "Edinburgh", 
    filename=None,
    repeat=True,
    fps=60,
    axis_scale=5,
    elev=10,
    azim=180,
    dist=10,
    floorSize=20,
    floorSquares=50,
    display_grid:bool = False,
    ):
        
    """
    add functionality to prove a list of frame names - e.g. it mist have same shape as positions
        Parameters:
            animation: Animation to be played (type is Animation)
            colour: must be one of the following
                    - None, defaults to 'red'
                    - colour name as a string, e.g. 'red' or 'green' (if name not recognised by matplotlib an error will be returned)
                    - a single (r,g,b) tuple, either 0.0 < r,g,b < 1.0 or 0 < r,g,b < 255
                    - or else a list, with length equal to number of skeletal joints, of rgb-tuples (r,g,b)
            filename: if a final name is provided the animation will be saved in video format
            fps: frame rate at which to play/save animation (probably won't be able to play at this rate but saved video will be okay)

            elev:
            azim:
            dist:


    """

    print("positions: ", positions.shape)

    # if animation is not None:
    #     parents = animation.parents
    #     positions = Animation.positions_global(animation)
    # elif (parents is None or positions is None ):
    #     raise AttributeError("Must provide either animation instance or both positions and parents")
    
    rootTranslation = positions[:,0,:]
    numFrames = positions.shape[0]
    numJoints = len(parents)

    if frame_info is not None and len(frame_info) == numFrames:
        display_titles = True
    else:
        display_titles = False

    # deal with provided colour(s)
    if skeleton is not None:
        colours = np.array(GetDefaultColours(skeleton))/255.0
    else: 
        colours = ['red'] * numJoints

    # create figure
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-axis_scale, axis_scale)
    ax.set_zlim3d( -axis_scale, axis_scale)
    ax.set_ylim3d(0, axis_scale)
    ax.set_box_aspect((1,1,0.5), zoom=1)
    
    if display_grid:
        ax.grid(display_grid)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        ax.set_axis_off()

    # initial camera position
    ax.view_init(elev=elev, azim=azim, vertical_axis='y')
   # ax.dist = dist

    # create floor
    xs = np.linspace(-floorSize, floorSize, floorSquares)
    zs = np.linspace(-floorSize, floorSize, floorSquares)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros(X.shape)
    wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='grey',lw=0.5)

    # display axes
    plt.plot([0,5], [0,0], [0,0], color='r', lw=4)
    plt.plot([0,0], [0,5], [0,0], color='g', lw=4)
    plt.plot([0,0], [0,0], [0,5], color='b', lw=4)
    
    
    # these are the lines (of type 'mpl_toolkits.mplot3d.art3d.Line3D') that will be drawn. 
    # one for the root trajectory and then one for each bone
    # the first 2 or 3 arguments are lists of the x, y (, and z) coordinates of the points the line is to pass trye, 
    lines = []
    lines.append(plt.plot(rootTranslation[:,0], np.zeros(numFrames), rootTranslation[:,2], lw=2, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])[0])
    lines.append([plt.plot([0,0], [0,0], [0,0], color=colours[j], lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for j in range(numJoints)])


    def animate(i):     

        if display_titles:
            ax.set_title("Frame: {:d}, {title}".format(i, title=frame_info[i]), y=0.95)
        else:
            ax.set_title("Frame: {:d}".format(i),  y=0.9) 
        for j in range(len(parents)):
            if parents[j] != -1:
                lines[1][j].set_data(np.array([[positions[i,j,0], positions[i,parents[j],0]],[positions[i,j,1],positions[i,parents[j],1]]]))
                lines[1][j].set_3d_properties(np.array([ positions[i,j,2],positions[i,parents[j],2]]))            
        return
        
    plt.tight_layout()
        
    ani = matplotlib.animation.FuncAnimation(fig,
        animate,
        np.arange(numFrames),
        interval=1000/fps,
        repeat=repeat)

    if filename != None:
        # Writer = matplotlib.animation.writers['ffmpeg']
        # writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(filename,fps=fps, bitrate=13934) # writer="ffmpeg"),0##, writer = FFwriter)
        ani.event_source.stop()
        del ani
        plt.close()    
    try:
        plt.show()
        plt.save()
    except AttributeError as e:
        pass

    return