# Adapted from Etienne Bennequin's https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/my_first_few_shot_classifier.ipynb

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Omniglot
from torchvision.models import resnet18
from tqdm import tqdm
from functools import partial

from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average

class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone: nn.Module):
        # Call parent constructor.
        super(PrototypicalNetwork, self).__init__()
        # Assign provided backbone.
        self.backbone = backbone

    def forward(self, support_images: torch.Tensor, support_labels: torch.Tensor, query_images: torch.Tensor) -> torch.Tensor:
        # Extract the features of support and query images.
        support_features = self.backbone.forward(support_images)
        query_features = self.backbone.forward(query_images)
        # Infer the number of different classes from the labels of the support set.
        n_way = len(torch.unique(support_labels))
        # Obtain a list of prototype features, where each is the mean of all instances of features corresponding to its corresponding label.
        prototypes = torch.cat(
            [ support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way) ]
        )
        # Compute the euclidean distance from queries to prototypes.
        distances = torch.cdist(query_features, prototypes)
        # Convert the euclidian distances to scores.
        scores = -distances
        # Return the resultant scores.
        return scores

def download_data(image_size):
    # Obtain the training set by downloading the Omniglot data.
    train_set = Omniglot(
        root="./data",
        background=True, # Select training set.
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        download=True,
    )
    # Obtain the testing set by downloading the Omniglot data.
    test_set = Omniglot(
        root="./data",
        background=False, # Select testing set.
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        ),
        download=True,
    )
    # Return both sets.
    return train_set, test_set

def get_labels_from_test_set(test_set):
    return [instance[1] for instance in test_set._flat_character_images]

def get_data_loader(test_set):
    n_way = 5  # Number of classes in a task.
    k_shot = 5  # Number of images per class in the support set.
    q_query = 10  # Number of images per class in the query set.
    evaluation_task_count = 100 # Number of evaluation tasks.
    # Provide the sample with a dataset containing a `get_labels` method.
    test_set.get_labels = partial(get_labels_from_test_set, test_set)
    test_sampler = TaskSampler(
        test_set, n_way=n_way, n_shot=k_shot, n_query=q_query, n_tasks=evaluation_task_count
    )
    # Create the data loader.
    return DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

def evaluate_on_one_task(support_images: torch.Tensor, support_labels: torch.Tensor, query_images: torch.Tensor, query_labels: torch.Tensor, model) -> [int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """
    # Return tuple of (number of correct predictions of query labels, total number of predictions).
    return (
        torch.max(
            model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
            .detach()
            .data,
            1,
        )[1]
        == query_labels.cuda()
    ).sum().item(), len(query_labels)


def evaluate(data_loader: DataLoader, model):
    total_predictions = 0
    correct_predictions = 0
    model.eval()
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):
            correct, total = evaluate_on_one_task(
                support_images, support_labels, query_images, query_labels, model
            )
            total_predictions += total
            correct_predictions += correct
    print(f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%")

def main():
    # Obtain the testing and training data sets.
    train_set, test_set = download_data(28)
    # Create the data loader.
    data_loader = get_data_loader(test_set)
    # Build our model from a pretrained backbone.
    convolutional_network = resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()
    model = PrototypicalNetwork(convolutional_network).cuda()
    # Load from our data loader.
    (
        example_support_images,
        example_support_labels,
        example_query_images,
        example_query_labels,
        example_class_ids,
    ) = next(iter(data_loader))
    # Evaluate the model.
    evaluate(data_loader, model)
    

if (__name__ == "__main__"):
    main()