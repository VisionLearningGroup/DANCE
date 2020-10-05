from .mydataset import ImageFolder
from .unaligned_data_loader import UnalignedDataLoader
from collections import Counter
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

def get_loader(source_path, target_path, evaluation_path, transforms,
               batch_size=32, return_id=False, balanced=False):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path],
                                return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False, return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms[evaluation_path],
                                   return_paths=True)
    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4)
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return source_loader, target_loader, test_loader, target_folder_train


def get_loader_class_inc(source_path, target_path, target_labeled_path, evaluation_path, transforms,
                         batch_size=32, return_id=False, balanced=False):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path],
                                return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False, return_id=return_id)
    target_folder_labeled = ImageFolder(os.path.join(target_labeled_path),
                                        transform=transforms[target_labeled_path],
                                        return_paths=False, return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms[evaluation_path],
                                   return_paths=True)
    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4)
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    target_labeled_loader = torch.utils.data.DataLoader(
        target_folder_labeled,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return source_loader, target_loader, target_labeled_loader, \
           test_loader, target_folder_train


def get_loader_balanced(source_path, target_path, evaluation_path, transforms, batch_size=32):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path])
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms[evaluation_path],
                                   return_paths=True)
    freq = Counter(source_folder.labels)
    class_weight = {x: 1.0 / freq[x] for x in freq}
    source_weights = [class_weight[x] for x in source_folder.labels]
    sampler = WeightedRandomSampler(source_weights, len(source_folder.labels))

    train_loader = UnalignedDataLoader()
    train_loader.initialize(source_folder, target_folder_train, batch_size, sampler=sampler)

    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return train_loader, test_loader
