from sklearn import svm
from dataset_extractor import load_dataset

def linear_svm_classifier(train_images, train_labels, test_images, test_labels):
    # initialize linear SVM classifier
    linear_clf = svm.SVC(kernel='linear')

    # fit data
    linear_clf.fit(train_images, train_labels)

    # calculate accuracy on test
    test_acc = linear_clf.score(test_images, test_labels)

    return test_acc

def rbf_svm_classifier(train_images, train_labels, test_images, test_labels):
    # initialize RBF SVM classifier
    rbf_clf = svm.SVC(kernel='rbf')

    # fit data
    rbf_clf.fit(train_images, train_labels)

    # calculate accuracy on test
    test_acc = rbf_clf.score(test_images, test_labels)

    return test_acc


if __name__ == "__main__":
    # load dataset
    train_images, train_labels, test_images, test_labels = load_dataset()

    # preform linear svm classification
    linear_svm_test_acc = linear_svm_classifier(train_images, train_labels, test_images, test_labels)
    print("Linear SVM Test Accuracy: " + str(linear_svm_test_acc))

    # preform rbf svm classification
    rbf_svm_test_acc = rbf_svm_classifier(train_images, train_labels, test_images, test_labels)
    print("RBF SVM Test Accuracy: " + str(rbf_svm_test_acc))