import sys
import re

def calculate_average_accuracy(file_path):
    """"
    Takes in a text file path as input and returns the average accuracy (mIoU, pixAcc) based on each
    line in the file content.

    Parameters
    ----------
    file_path: str
        The given path to the text file
    
    Returns
    -------
    tuple
        A tuple containing the average mIoU and the average pixel accuracy
    """
    file_lines = get_file_content(file_path)
    avg_miou = 0.0
    avg_pix_acc = 0.0
    for line in file_lines:
        if line != "":
            acc = get_miou_pixacc(line)
            if not acc:
                print("Incorrect Format: No values for mIoU or pixAcc found in this file")
                sys.exit(1)
            avg_miou += acc[0]
            avg_pix_acc += acc[1]
    return (avg_miou / len(file_lines), avg_pix_acc / len(file_lines))

def get_file_content(file_path):
    """
    Takes in a text file path as input and returns the file content of the text file as a list of
    strings. The list of strings represents the content of the file split by new line characters.

    Parameters
    ----------
    file_path: str
        The given path to the text file
    
    Returns
    -------
    list
        A list of strs that represent each line from the content of the text file
    """
    file = open(file_path, "r")
    content = file.read()
    return content.split('\n')

def get_miou_pixacc(line):
    """
    Takes in a string representing a line and returns the mIoU value and the pixel accuracy
    value found within that line. If the mIoU or the pixel accuracy does not exist in this line, 
    then returns None.

    Parameters
    ----------
    line: str
        The given line containing some content
    
    Returns
    -------
    tuple
        Returns a tuple containing the mIoU value and the pixel accuracy in the form (mIoU, pixAcc)
        or returns None if it doesn't exist
    """
    pixel_acc_match = re.search(r'pixAcc: [\d.]+', line)
    pixel_acc = None
    if pixel_acc_match:
        pixel_acc = float(pixel_acc_match.group(0).split(' ')[-1])
    miou_match = re.search(r'mIoU: [\d.]+', line)
    miou = None
    if miou_match:
        miou = float(miou_match.group(0).split(' ')[-1])
    if not miou or not pixel_acc:
        return None
    return (miou, pixel_acc)

def main():
    args = sys.argv
    if len(args) != 2:
        print("Argument Error: incorrect number of arguments")
        sys.exit(1)
    file_path = args[1]
    accuracy = calculate_average_accuracy(file_path)
    print(accuracy)

if __name__ == "__main__":
	main()