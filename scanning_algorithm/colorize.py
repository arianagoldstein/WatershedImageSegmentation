import random

# function to print the matrix from the given input file
def colorize(file_name):
    """
    Function to print the given matrix in color, assigning a unique color to each unique number
    :param file_name: file that contains matrix to print
    :return: none
    """
    number_set = set()
    # reading in file
    with open(file_name, "r") as f:
        lines = f.readlines()
        matrix = []
        for line in lines:
            row = [int(x) for x in line.split()]
            # maintaining a set to represent the unique numbers in the file
            for num in row:
                number_set.add(int(num))
            matrix.append(row)

    # generate n random colors where n is the number of unique numbers
    color_list = []
    for i in range(len(number_set)):
        unique = False
        while not unique:
            color = (random.randint(0, 1), random.randint(30, 37))
            if color not in color_list:
                unique = True
                color_list.append(color)

    num_color_pairs = dict(zip(number_set, color_list))

    # printing colorized matrix
    for row in matrix:
        for elem in row:
            color_style = num_color_pairs[elem][0]
            color_value = num_color_pairs[elem][1]
            color = str("\033[" + str(color_style) + ";" + str(color_value) + "m")
            print(color, str(elem), end="")
        print("")


if __name__ == "__main__":
    colorize('input.txt')