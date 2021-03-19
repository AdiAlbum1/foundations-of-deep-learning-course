import pickle

DATASET_BATCHES = 5

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def extract_dataset():
    all_images = []
    all_labels = []
    for i in range(1, DATASET_BATCHES + 1):
        curr_dataset_batch = unpickle("dataset/cifar-10-batches-py/data_batch_" + str(i))
        all_images.extend(curr_dataset_batch[b'data'])
        all_labels.extend(curr_dataset_batch[b'labels'])

    return all_images, all_labels

if __name__ == "__main__":
    all_images, all_labels = extract_dataset()