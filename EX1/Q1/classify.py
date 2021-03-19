from sklearn import svm
from dataset_extractor import load_dataset

if __name__ == "__main__":
    # load dataset
    train_images, train_labels, test_images, test_labels = load_dataset()

    # initialize linear SVM classifier
    linear_clf = svm.SVC(kernel='linear')

    # fit data
    linear_clf.fit(train_images, train_labels)

    # calculate accuracy on test
    test_acc = linear_clf.score(test_images, test_labels)
    print("TEST ACCURACY: " + str(test_acc))