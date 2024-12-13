import os
import torch
import clip
from tqdm import tqdm
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

SAVED_DATASETS_PATH = ""

def get_features(dataset, batch_size=100):
    """
    Extracts image embeddings/features from a dataset using the CLIP model.
    ---
    dataset: torch.utils.data.Dataset
    batch_size: int

    returns:
    all_features: numpy array of shape (N, 512)
    all_labels: numpy array of shape (N,)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, preprocess = clip.load('ViT-B/32', device=device)

    if not hasattr(dataset, 'transform') or dataset.transform is None:
        dataset.transform = preprocess

    all_features = []
    all_labels = []

    dataloader = DataLoader(dataset, batch_size=batch_size)
    for images, labels in tqdm(dataloader, desc="Extracting features"):
        with torch.no_grad():
            images = images.to(device)
            features = model.encode_image(images)
            all_features.append(features.cpu())
            all_labels.append(labels)

    all_features = torch.cat(all_features).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return all_features, all_labels

def get_CIFAR10_features():
    """
    Extracts image embeddings/features from the CIFAR10 dataset using the CLIP model.

    ---
    returns:
    train_features: numpy array of shape (50000, 512)
    train_labels: numpy array of shape (50000,)
    test_features: numpy array of shape (10000, 512)
    test_labels: numpy array of shape (10000,)
    """
    root = os.path.expanduser("~/.cache")
    train = CIFAR10(root, download=True, train=True)
    test = CIFAR10(root, download=True, train=False)

    train_features, train_labels = None, None
    test_features, test_labels = None, None

    # Extracted features file paths names
    train_feature_file = SAVED_DATASETS_PATH + 'CIFAR10_CLIP_image_train.npz'
    test_feature_file = SAVED_DATASETS_PATH + 'CIFAR10_CLIP_image_test.npz'

    # Load extracted features if possible, else call CLIP and save to disk
    if os.path.exists(train_feature_file) and os.path.exists(test_feature_file):
        train_data = np.load(train_feature_file)
        train_features = train_data['features']
        train_labels = train_data['labels']

        test_data = np.load(test_feature_file)
        test_features = test_data['features']
        test_labels = test_data['labels']
        print("Loaded features from disk.")
    else:
        train_features, train_labels = get_features(train)
        test_features, test_labels = get_features(test)

        np.savez(train_feature_file, features=train_features, labels=train_labels)
        np.savez(test_feature_file, features=test_features, labels=test_labels)

    return train_features, train_labels, test_features, test_labels

def get_CIFAR100_features():
    """
    Extracts image embeddings/features from the CIFAR100 dataset using the CLIP model.
    ---
    returns:
    train_features: numpy array of shape (50000, 512)
    train_labels: numpy array of shape (50000,)
    test_features: numpy array of shape (10000, 512)
    test_labels: numpy array of shape (10000,)
    """
    root = os.path.expanduser("~/.cache")
    train = CIFAR100(root, download=True, train=True)
    test = CIFAR100(root, download=True, train=False)

    train_features, train_labels = None, None
    test_features, test_labels = None, None

    # Extracted features file paths names
    train_feature_file = SAVED_DATASETS_PATH + 'CIFAR100_CLIP_image_train.npz'
    test_feature_file = SAVED_DATASETS_PATH + 'CIFAR100_CLIP_image_test.npz'

    # saving to file names
    print(f"Extracting features from CIFAR100 dataset")

    # Load extracted features if possible, else call CLIP and save to disk
    if os.path.exists(train_feature_file) and os.path.exists(test_feature_file):
        train_data = np.load(train_feature_file)
        train_features = train_data['features']
        train_labels = train_data['labels']

        test_data = np.load(test_feature_file)
        test_features = test_data['features']
        test_labels = test_data['labels']
        print("Loaded previously extracted features from disk.")
    else:
        train_features, train_labels = get_features(train)
        test_features, test_labels = get_features(test)

        np.savez(train_feature_file, features=train_features, labels=train_labels)
        np.savez(test_feature_file, features=test_features, labels=test_labels)

    return train_features, train_labels, test_features, test_labels

def get_MNIST_features():
    """
    Extracts image embeddings/features from the MNIST dataset using the CLIP model.

    ---
    returns:
    train_features: numpy array of shape (60000, 512)
    train_labels: numpy array of shape (60000,)
    test_features: numpy array of shape (10000, 512)
    test_labels: numpy array of shape (10000,)
    """
    root = os.path.expanduser("~/.cache")
    train = MNIST(root, download=True, train=True)
    test = MNIST(root, download=True, train=False)

    train_features, train_labels = None, None
    test_features, test_labels = None, None

    # Extracted features file paths names
    train_feature_file = SAVED_DATASETS_PATH + 'MNIST_CLIP_image_train.npz'
    test_feature_file = SAVED_DATASETS_PATH + 'MNIST_CLIP_image_test.npz'

    # Load extracted features if possible, else call CLIP and save to disk
    if os.path.exists(train_feature_file) and os.path.exists(test_feature_file):
        train_data = np.load(train_feature_file)
        train_features = train_data['features']
        train_labels = train_data['labels']

        test_data = np.load(test_feature_file)
        test_features = test_data['features']
        test_labels = test_data['labels']
        print("Loaded features from disk.")
    else:
        train_features, train_labels = get_features(train)
        test_features, test_labels = get_features(test)

        np.savez(train_feature_file, features=train_features, labels=train_labels)
        np.savez(test_feature_file, features=test_features, labels=test_labels)

    return train_features, train_labels, test_features, test_labels


### OpenCLIP

import open_clip

def get_features_OpenCLIP(dataset, model_name='ViT-H-14-378-quickgelu', pretrained='dfn5b', batch_size=100):
    """
    Extracts image embeddings/features from a dataset using an OpenCLIP model.
    ---
    dataset: torch.utils.data.Dataset
    model_name: str, the OpenCLIP model name (e.g., 'ViT-B-32')
    pretrained: str, the OpenCLIP pretrained weights (e.g., 'laion2b_s34b_b79k')
    batch_size: int

    returns:
    all_features: numpy array of shape (N, D) where D depends on the model (512 for ViT-B-32)
    all_labels: numpy array of shape (N,)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()

    # Ensure dataset uses the correct preprocessing transform
    if not hasattr(dataset, 'transform') or dataset.transform is None:
        dataset.transform = preprocess

    dataloader = DataLoader(dataset, batch_size=batch_size)

    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Extracting features with OpenCLIP"):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device=='cuda')):
            images = images.to(device)
            features = model.encode_image(images)
            # Normalize features if you like (optional)
            # features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu())
            all_labels.append(labels)

    all_features = torch.cat(all_features).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return all_features, all_labels

def get_CIFAR10_features_OpenCLIP( model_name='ViT-H-14-378-quickgelu', pretrained='dfn5b', batch_size=100):
    """
    Extracts image embeddings/features from the CIFAR10 dataset using an OpenCLIP model.

    ---
    model_name: str, the OpenCLIP model name
    pretrained: str, the OpenCLIP pretrained weights
    batch_size: int

    returns:
    train_features: numpy array of shape (50000, 1024) D = 1024 for default model
    train_labels: numpy array of shape (50000,)
    test_features: numpy array of shape (10000, 1024) D = 1024 for default model
    test_labels: numpy array of shape (10000,)
    """
    root = os.path.expanduser("~/.cache")
    train = CIFAR10(root, download=True, train=True)
    test = CIFAR10(root, download=True, train=False)

    # Extracted features file paths
    train_feature_file = os.path.join(SAVED_DATASETS_PATH, f'CIFAR10_OpenCLIP_{model_name}_train.npz')
    test_feature_file = os.path.join(SAVED_DATASETS_PATH, f'CIFAR10_OpenCLIP_{model_name}_test.npz')

    if os.path.exists(train_feature_file) and os.path.exists(test_feature_file):
        train_data = np.load(train_feature_file)
        train_features = train_data['features']
        train_labels = train_data['labels']

        test_data = np.load(test_feature_file)
        test_features = test_data['features']
        test_labels = test_data['labels']
        print("Loaded OpenCLIP features from disk.")
    else:
        train_features, train_labels = get_features_OpenCLIP(train, model_name, pretrained, batch_size)
        test_features, test_labels = get_features_OpenCLIP(test, model_name, pretrained, batch_size)

        np.savez(train_feature_file, features=train_features, labels=train_labels)
        np.savez(test_feature_file, features=test_features, labels=test_labels)
        print("Extracted and saved OpenCLIP features to disk.")

    return train_features, train_labels, test_features, test_labels

def get_CIFAR100_features_OpenCLIP( model_name='ViT-H-14-378-quickgelu', pretrained='dfn5b', batch_size=100):
    """
    Extracts image embeddings/features from the CIFAR100 dataset using an OpenCLIP model.

    ---
    model_name: str, the OpenCLIP model name
    pretrained: str, the OpenCLIP pretrained weights
    batch_size: int

    returns:
    train_features: numpy array of shape (50000, 1024) D = 1024 for default model
    train_labels: numpy array of shape (50000,)
    test_features: numpy array of shape (10000, 1024) D = 1024 for default model
    test_labels: numpy array of shape (10000,)
    """
    root = os.path.expanduser("~/.cache")
    train = CIFAR100(root, download=True, train=True)
    test = CIFAR100(root, download=True, train=False)

    # Extracted features file paths
    train_feature_file = os.path.join(SAVED_DATASETS_PATH, f'CIFAR100_OpenCLIP_{model_name}_train.npz')
    test_feature_file = os.path.join(SAVED_DATASETS_PATH, f'CIFAR100_OpenCLIP_{model_name}_test.npz')

    if os.path.exists(train_feature_file) and os.path.exists(test_feature_file):
        train_data = np.load(train_feature_file)
        train_features = train_data['features']
        train_labels = train_data['labels']

        test_data = np.load(test_feature_file)
        test_features = test_data['features']
        test_labels = test_data['labels']
        print("Loaded OpenCLIP features from disk.")
    else:
        train_features, train_labels = get_features_OpenCLIP(train, model_name, pretrained, batch_size)
        test_features, test_labels = get_features_OpenCLIP(test, model_name, pretrained, batch_size)

        np.savez(train_feature_file, features=train_features, labels=train_labels)
        np.savez(test_feature_file, features=test_features, labels=test_labels)
        print("Extracted and saved OpenCLIP features to disk.")

    return train_features, train_labels, test_features, test_labels

def get_MNIST_features_OpenCLIP( model_name='ViT-H-14-378-quickgelu', pretrained='dfn5b', batch_size=100):
    """
    Extracts image embeddings/features from the MNIST dataset using an OpenCLIP model.

    ---
    model_name: str, the OpenCLIP model name
    pretrained: str, the OpenCLIP pretrained weights
    batch_size: int

    returns:
    train_features: numpy array of shape (60000, 1024) D = 1024 for default model
    train_labels: numpy array of shape (60000,)
    test_features: numpy array of shape (10000, 1024) D = 1024 for default model
    test_labels: numpy array of shape (10000,)
    """
    root = os.path.expanduser("~/.cache")
    train = MNIST(root, download=True, train=True)
    test = MNIST(root, download=True, train=False)

    # Extracted features file paths
    train_feature_file = os.path.join(SAVED_DATASETS_PATH, f'MNIST_OpenCLIP_{model_name}_train.npz')
    test_feature_file = os.path.join(SAVED_DATASETS_PATH, f'MNIST_OpenCLIP_{model_name}_test.npz')

    if os.path.exists(train_feature_file) and os.path.exists(test_feature_file):
        train_data = np.load(train_feature_file)
        train_features = train_data['features']
        train_labels = train_data['labels']

        test_data = np.load(test_feature_file)
        test_features = test_data['features']
        test_labels = test_data['labels']
        print("Loaded OpenCLIP features from disk.")
    else:
        train_features, train_labels = get_features_OpenCLIP(train, model_name, pretrained, batch_size)
        test_features, test_labels = get_features_OpenCLIP(test, model_name, pretrained, batch_size)

        np.savez(train_feature_file, features=train_features, labels=train_labels)
        np.savez(test_feature_file, features=test_features, labels=test_labels)
        print("Extracted and saved OpenCLIP features to disk.")

    return train_features, train_labels, test_features, test_labels