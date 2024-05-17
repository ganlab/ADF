import argparse
import glob
import numpy as np
from sklearn.metrics import confusion_matrix
import os

def calculate_class_accuracies(labels, preds, num_classes):
    # Generate confusion matrix
    cm = confusion_matrix(labels, preds, labels=range(num_classes))

    # Extract the diagonal from the confusion matrix, representing the correct classifications for each class
    true_positive = np.diag(cm)

    # Calculate the total number of actual instances for each class (by summing rows of the confusion matrix)
    actual_positive = np.sum(cm, axis=1)

    # Create a float type output array for the output of np.divide
    output_array = np.zeros_like(true_positive, dtype=np.float64)

    # Calculate class accuracies separately, using np.divide with an 'ignore' parameter to avoid division by zero
    class_accuracies = np.divide(true_positive, actual_positive, out=output_array, where=actual_positive != 0)

    return class_accuracies

def parse_number_list(number_list):
    try:
        return list(map(float, number_list.strip('[]').split(',')))
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid number list format: {number_list}. Expected format: [num1,num2,...]")


def main(args):
    base_path = args.path

    pcd_names = os.listdir(base_path)
    for pcd_name in pcd_names:
        pcd_path = os.path.join(base_path, pcd_name, pcd_name + '.npy')
        # Load point cloud data
        point_cloud_data = np.load(pcd_path)
        k_names = [f for f in os.listdir(os.path.join(base_path, pcd_name, 'mask')) if
                   os.path.isdir(os.path.join(os.path.join(base_path, pcd_name, 'mask'), f))]
        for k in k_names:
            print(pcd_name + '_' + k)
            preds_path_folder = os.path.join(base_path, pcd_name, 'mask', str(k))
            file_pattern = os.path.join(preds_path_folder, 'label.txt')
            file_paths = glob.glob(file_pattern)
            preds_names = [os.path.basename(file_path) for file_path in file_paths]
            # Load prediction label data
            for preds_name in preds_names:
                preds_path = os.path.join(preds_path_folder, preds_name)
                preds = np.loadtxt(preds_path)

                # Ensure labels and preds are numpy arrays of the same length
                labels = point_cloud_data[:,6:]
                labels = np.array(labels)
                preds = np.array(preds) - 1

                # Generate confusion matrix
                cm = confusion_matrix(labels, preds)

                # Calculate OA (Overall Accuracy)
                OA = np.diag(cm).sum() / cm.sum()

                # Calculate Accuracy for each class
                with np.errstate(divide='ignore', invalid='ignore'):
                    individual_acc = np.diag(cm) / cm.sum(axis=1)
                    individual_acc = np.nan_to_num(individual_acc)  # Convert NaNs to 0, which occurs when a class is not present in the denominator

                # Calculate mAcc (Mean Accuracy)
                mAcc = np.mean(individual_acc)

                # Calculate IoU for each class and mIoU (Mean IoU)
                individual_iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
                mIoU = np.nanmean(individual_iou)  # Use nanmean to avoid NaN issues caused by zero values

                # Output results
                print(f"Overall Accuracy (OA): {OA * 100:.1f}%")
                print(f"Mean Accuracy (mAcc): {mAcc * 100:.1f}%")
                print(f"Mean IoU (mIoU): {mIoU * 100:.1f}%")

                # Assume you have the actual labels and predicted labels, along with the number of classes
                num_classes = 13
                # Call function to obtain accuracies for each class
                class_accuracies = calculate_class_accuracies(labels, preds, num_classes)
                # Assuming print each class accuracy
                category_names = {
                    0: "ceiling",
                    1: "floor",
                    2: "wall",
                    3: "beam",
                    4: "column",
                    5: "window",
                    6: "door",
                    7: "table",
                    8: "chair",
                    9: "sofa",
                    10: "bookcase",
                    11: "board",
                    12: "clutter"
                }
                for i, acc in enumerate(class_accuracies):
                    if i in category_names:
                        print(f"Accuracy for class {i} ({category_names[i]}): {acc * 100:.1f}%")
                    else:
                        print(f"Accuracy for class {i}: {acc * 100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--path", type=str, help="Base path", default='./examples')

    args = parser.parse_args()
    main(args)