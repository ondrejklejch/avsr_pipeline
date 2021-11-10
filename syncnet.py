from collections import namedtuple
import numpy as np
import python_speech_features
from scipy.signal import medfilt
from syncnet_python.SyncNetModel import S
import time
import torch

SyncNet = namedtuple('SyncNet', ['model', 'device'])

def load_syncnet(device):
    tstamp = time.time()

    print('[SyncNet] loading with', device)
    model = S().cuda(device)
    model_state = model.state_dict();

    loaded_state = torch.load('syncnet_python/data/syncnet_v2.model', map_location=lambda storage, loc: storage);
    for name, param in loaded_state.items():
        model_state[name].copy_(param);

    model.eval()
    print('[SyncNet] finished loading (%.4f sec)' % (time.time() - tstamp))

    return SyncNet(model, device)


def find_talking_segments(syncnet, facetrack, threshold, min_speech_duration, max_pause_duration):
    with torch.no_grad():
        fconf, fconfm = get_syncnet_scores(syncnet, facetrack)
        segments = get_syncnet_segments(fconfm, threshold, min_speech_duration, max_pause_duration)
        
        for start, end in segments:
            yield facetrack.trim(start, end)


def get_syncnet_scores(syncnet, facetrack):
    frames = np.stack(facetrack.frames, axis=3)
    frames = np.expand_dims(frames, axis=0)
    frames = np.transpose(frames, (0,3,4,1,2))
    
    mfcc = zip(*python_speech_features.mfcc(facetrack.audio, facetrack.audio_sample_rate))
    mfcc = np.stack([np.array(i) for i in mfcc])
    mfcc = np.expand_dims(np.expand_dims(mfcc, axis=0),axis=0)
    
    frames_length = frames.shape[2]
    audio_length = mfcc.shape[-1] // 4
    length = min(frames_length, audio_length) - 5
    
    batch_size = 20
    visual_feats = []
    audio_feats = []
    
    for start in range(0, length, batch_size):
        end = min(start + batch_size, length)
        
        frames_batch = [frames[:,:,vframe:vframe+5,:,:] for vframe in range(start, end)]
        frames_batch = torch.cat([torch.from_numpy(f.astype(float)).float() for f in frames_batch], 0)
        visual_feats.append(syncnet.model.forward_lip(frames_batch.cuda(syncnet.device)).cpu())

        mfcc_batch = [mfcc[:,:,:,vframe*4:vframe*4+20] for vframe in range(start, end)]
        mfcc_batch = torch.cat([torch.from_numpy(f.astype(float)).float() for f in mfcc_batch], 0)
        audio_feats.append(syncnet.model.forward_aud(mfcc_batch.cuda(syncnet.device)).cpu())

    visual_feats = torch.cat(visual_feats, axis=0)
    audio_feats = torch.cat(audio_feats, axis=0)
    dists = calc_pdist(visual_feats, audio_feats, vshift=15)
    mdist = torch.mean(torch.stack(dists, 1), 1)
    minval, minidx = torch.min(mdist, 0)
    fdist = np.stack([dist[minidx].numpy() for dist in dists])
    fconf = torch.median(mdist).numpy() - fdist
    fconfm = medfilt(fconf, kernel_size=25)

    return fconf, fconfm


def get_syncnet_segments(fconfm, threshold, min_speech_duration, max_pause_duration):
    inside = False
    start = None
    segments = []   
    for i, is_synchronised in enumerate(list(fconfm > threshold) + [True, True, True, True, True, False]):
        if inside:
            if not is_synchronised:
                if (i - start + 1) >= min_speech_duration:
                    if len(segments) == 0 or (start - segments[-1][1]) > max_pause_duration:
                        segments.append((start, i))
                    else:
                        segments[-1] = (segments[-1][0], i)
                    
                start = None
                inside = False
        elif is_synchronised:
            start = i
            inside = True
            
    return segments


def calc_pdist(feat1, feat2, vshift=15):
    win_size = vshift*2+1
    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))
    dists = []
    for i in range(0,len(feat1)):
        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))
    return dists
