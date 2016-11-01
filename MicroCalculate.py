def calculateMicroValue(y_pred, y_true, labels=None):
    if labels is None:
        labels = [0, 1]
    all_label_result = {}
    for label in labels:
        all_label_result[label] = calculateF(y_pred, y_true, label)
    print "Corresponding result -->", all_label_result
    tp = fp = fn = 0.0
    for label in all_label_result:
        tp += all_label_result[label][0]
        fp += all_label_result[label][1]
        fn += all_label_result[label][2]
    p = tp / (tp + fp + 0.00001)
    r = tp / (tp + fn + 0.00001)
    f = 2 * p * r / (p + r + 0.00001)
    print "for all label", str(labels), "\t p=", p, "\tr=", r, "\tf=", f
    for label in all_label_result:
        tp = all_label_result[label][0]
        fp = all_label_result[label][1]
        fn = all_label_result[label][2]
        p = tp / (tp + fp + 0.00001)
        r = tp / (tp + fn + 0.00001)
        f = 2 * p * r / (p + r + 0.00001)
        # print "for label", label, "\t p=", p, "\tr=", r, "\tf=", f
    return p, r, f


def calculateF(y_pred, y_true, label):
    tp = fp = fn = 0.0
    for left, right in zip(y_pred, y_true):
        if label == left and right == label:
            tp += 1
        if left == label and right != label:
            fp += 1
        if left != label and right == label:
            fn += 1
    return [tp, fp, fn]

# import numpy
# samples = 10
# y_pred = numpy.random.randint(low=0, high=3, size=(samples, ))
# y_true = numpy.random.randint(low=0, high=3, size=(samples, ))
# calculateMicroValue(y_pred, y_true)
