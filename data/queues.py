# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
from data.datasets import UserEpisodicORBITDataset, ObjectEpisodicORBITDataset
from data.samplers import TaskSampler

class DatasetQueue:
    """
    Class for a queue of tasks sampled from UserEpisodicORIBTDataset/ObjectEpisodicORBITDataset.

    """
    def __init__(self, tasks_per_user, shuffle,  test_mode):
        """
        Creates instance of DatasetQueue.
        :param tasks_per_user: (int) Number of tasks per user to add to the queue.
        :param shuffle: (bool) If True, shuffle tasks, else do not shuffled.
        :param test_mode: (bool) If True, only return target set for first task per user.
        :return: Nothing.
        """
        self.tasks_per_user = tasks_per_user
        self.shuffle = shuffle
        self.test_mode = test_mode
        self.num_workers = 4 if self.test_mode else 8
        self.num_users = None
        self.collate_fn = self.squeeze

    def squeeze(self, batch):
        #assumes batch_size = 1
        squeezed_batch = {} 
        for k,v in batch[0].items():
            if isinstance(v, torch.Tensor):
                squeezed_batch[k] = v.squeeze(0)
            elif isinstance(v, list):
                if all(isinstance(b, np.ndarray) for b in v):
                    squeezed_batch[k] = v
                elif all(isinstance(b, torch.Tensor) for b in v):
                    squeezed_batch[k] = [b.squeeze(0) for b in v]
            elif isinstance(v, np.ndarray):
                squeezed_batch[k] = v
        
        return squeezed_batch
    
    def get_num_users(self):
        return self.num_users
    
    def get_cluster_classes(self):
        return self.dataset.cluster_classes
    
    def get_tasks(self):
        return torch.utils.data.DataLoader(
                dataset=self.dataset,
                pin_memory=False,
                num_workers=self.num_workers,
                sampler=TaskSampler(self.tasks_per_user, self.num_users, self.shuffle, self.test_mode),
                collate_fn=self.collate_fn
                ) 
    
class UserEpisodicDatasetQueue(DatasetQueue):
    def __init__(self, root, way_method, object_cap, shot_method, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, tasks_per_user, test_mode, with_cluster_labels, with_caps, shuffle):
        DatasetQueue.__init__(self, tasks_per_user, shuffle, test_mode)
        self.dataset = UserEpisodicORBITDataset(root, way_method, object_cap, shot_method, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, test_mode, with_cluster_labels, with_caps)
        self.num_users = self.dataset.num_users
    
class ObjectEpisodicDatasetQueue(DatasetQueue):
    def __init__(self, root, way_method, object_cap, shot_method, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, tasks_per_user, test_mode, with_cluster_labels, with_caps, shuffle):
        DatasetQueue.__init__(self, tasks_per_user, shuffle, test_mode)
        self.dataset = ObjectEpisodicORBITDataset(root, way_method, object_cap, shot_method, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, test_mode, with_cluster_labels, with_caps)
        self.num_users = self.dataset.num_users    
