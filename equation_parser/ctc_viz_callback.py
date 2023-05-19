import numpy as np
import itertools
from tensorflow import keras


def decode_batch(test_func, tokenizer, word_batch):
    """
    Takes the Batch of Predictions and decodes the Predictions by Best Path Decoding and Returns the Output
    """
    out = test_func([word_batch])[
        0]  # returns the predicted output matrix of the model
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = tokenizer.sequences_to_texts([out_best])[0]
        ret.append(outstr)
    return ret


def accuracies(actual_labels, predicted_labels, is_train):
    """
    Takes a List of Actual Outputs, predicted Outputs and returns their accuracy and letter accuracy across
    all the labels in the list
    """
    accuracy = 0
    letter_acc = 0
    letter_cnt = 0
    count = 0
    for i in range(len(actual_labels)):
        predicted_output = predicted_labels[i]
        actual_output = actual_labels[i]
        count += 1
        for j in range(min(len(predicted_output), len(actual_output))):
            if predicted_output[j] == actual_output[j]:
                letter_acc += 1
        letter_cnt += max(len(predicted_output), len(actual_output))
        if actual_output == predicted_output:
            accuracy += 1
    final_accuracy = np.round((accuracy/len(actual_labels))*100, 2)
    final_letter_acc = np.round((letter_acc/letter_cnt)*100, 2)
    return final_accuracy, final_letter_acc


class CtcVizCallback(keras.callbacks.Callback):
    """
    The Custom Callback created for printing the Accuracy and Letter Accuracy Metrics at the End of Each Epoch
    """

    def __init__(self, test_func, text_img_gen, is_train, acc_compute_batches, tokenizer):
        self.test_func = test_func
        self.text_img_gen = text_img_gen
        # used to indicate whether the callback is called to for Train or Validation Data
        self.is_train = is_train
        # Number of Batches for which the metrics are computed typically equal to steps/epoch
        self.acc_batches = acc_compute_batches
        self.tokenizer = tokenizer

    def show_accuracy_metrics(self, num_batches):
        """
        Calculates the accuracy and letter accuracy for each batch of inputs, 
        and prints the avarage accuracy and letter accuracy across all the batches
        """
        accuracy = 0
        letter_accuracy = 0
        batches_cnt = num_batches
        while batches_cnt > 0:
            # Gets the next batch from the Data generator
            word_batch = next(self.text_img_gen)[0]
            decoded_res = decode_batch(
                self.test_func, self.tokenizer, word_batch['img_input'])
            actual_res = word_batch['source_str']
            acc, let_acc = accuracies(actual_res, decoded_res, self.is_train)
            accuracy += acc
            letter_accuracy += let_acc
            batches_cnt -= 1
        accuracy = accuracy/num_batches
        letter_accuracy = letter_accuracy/num_batches
        if self.is_train:
            print("Train Average Accuracy of "+str(num_batches) +
                  " Batches: ", np.round(accuracy, 2), " %")
            print("Train Average Letter Accuracy of "+str(num_batches) +
                  " Batches: ", np.round(letter_accuracy, 2), " %")
        else:
            print("Validation Average Accuracy of "+str(num_batches) +
                  " Batches: ", np.round(accuracy, 2), " %")
            print("Validation Average Letter Accuracy of "+str(num_batches) +
                  " Batches: ", np.round(letter_accuracy, 2), " %")

    def on_epoch_end(self, epoch, logs={}):
        self.show_accuracy_metrics(self.acc_batches)
