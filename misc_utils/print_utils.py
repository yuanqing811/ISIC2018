import numpy as np
import sys
import logging
import sys
from paths import root_dir

class PrintColors:

    GREEN = "\033[0;32m"
    BLUE = "\033[1;34m"
    RED = "\033[1;31m"

    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END_COLOR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def on_aws():
    if 'ubuntu' in root_dir:
        return True
    return False


def log_variable(var_name, var_value):
    print('{: <20} : {}'.format(var_name, var_value))


def print_confusion_matrix(cm, labels):
    """pretty print for confusion matrixes"""

    columnwidth = max([len(x) for x in labels] + [12])
    # Print header
    print()
    first_cell = "True\Pred"
    print("|%{0}s|".format(columnwidth - 2) % first_cell, end="")
    for label in labels:
        print("%{0}s|".format(columnwidth -1) % label, end="")
    print()

    first_cell = "-------"
    print("|%{0}s|".format(columnwidth-2) % first_cell, end="")
    for _ in labels:
        print("%{0}s|".format(columnwidth-1) % first_cell, end="")
    print()

    # Print rows
    for i, label1 in enumerate(labels):
        print("|%{0}s|".format(columnwidth - 2) % label1, end="")
        for j in range(len(labels)):
            cell = "%{0}.2f|".format(columnwidth-1) % cm[i, j]
            if i == len(labels) - 1 or j == len(labels) - 1:
                cell = "%{0}d|".format(columnwidth-1) % cm[i, j]
                if i == j:
                    print("%{0}s|".format(columnwidth-1) % ' ', end="")
                else:
                    print(PrintColors.BLUE + cell + PrintColors.END_COLOR, end="")
            elif i == j:
                print(PrintColors.GREEN + cell + PrintColors.END_COLOR, end="")
            else:
                print(PrintColors.RED + cell + PrintColors.END_COLOR, end="")

        print()


def print_precision_recall(precision, recall, labels):
    columnwidth = max([len(x) for x in labels] + [12])
    # Print header
    print()
    first_cell = " "
    print("|%{0}s|".format(columnwidth-2) % first_cell, end="")
    for label in labels:
        print("%{0}s|".format(columnwidth-1) % label, end="")
    print("%{0}s|".format(columnwidth-1) % 'MEAN', end="")
    print()

    first_cell = "-------"
    print("|%{0}s|".format(columnwidth-2) % first_cell, end="")
    for _ in labels:
        print("%{0}s|".format(columnwidth-1) % first_cell, end="")
    print("%{0}s|".format(columnwidth-1) % first_cell, end="")
    print()

    # print precision
    print("|%{0}s|".format(columnwidth-2) % 'precision', end="")
    for j in range(len(labels)):
        cell = "%{0}.3f|".format(columnwidth-1) % precision[j]
        print(PrintColors.GREEN + cell + PrintColors.END_COLOR, end="")

    cell = "%{0}.3f|".format(columnwidth-1) % np.mean(precision)
    print(PrintColors.BLUE + cell + PrintColors.END_COLOR, end="")

    print()

    # print recall
    print("|%{0}s|".format(columnwidth-2) % 'recall', end="")
    for j in range(len(labels)):
        cell = "%{0}.3f|".format(columnwidth-1) % recall[j]
        print(PrintColors.GREEN + cell + PrintColors.END_COLOR, end="")

    cell = "%{0}.3f|".format(columnwidth-1) % np.mean(recall)
    print(PrintColors.BLUE + cell + PrintColors.END_COLOR, end="")

    print('')


class Tee(object):
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
        self.__missing_method_name = None # Hack!

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        self.__missing_method_name = name # Could also be a property
        return getattr(self, '__methodmissing__')

    def __methodmissing__(self, *args, **kwargs):
        # Emit method call to the log copy
        callable2 = getattr(self.stream2, self.__missing_method_name)
        callable2(*args, **kwargs)

        # Emit method call to stdout (stream 1)
        callable1 = getattr(self.stream1, self.__missing_method_name)
        return callable1(*args, **kwargs)
