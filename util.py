import os

if __name__ == '__main__':
    # Get a list of all files and directories within the specified path
    attribute_director_path = 'sandbox/train_model_3/valid/labels'
    all_entries = os.listdir(attribute_director_path)

    # Filter the list to include only files
    file_names = [entry for entry in all_entries if os.path.isfile(os.path.join(attribute_director_path, entry))]

    sorted_file_names = sorted(file_names)

    altitude = 100.0
    for attr_file_name in sorted_file_names:
        # Read all lines
        with open(os.path.join(attribute_director_path, attr_file_name), "r") as f:
            lines = f.readlines()

        # Modify line by line
        new_lines = []
        for line in lines:
            # *rest, alt = line.strip().split()
            # label = ' '.join(rest)
            new_line =  f'{line.strip()} {altitude}\n'
            new_lines.append(new_line)

        # Write back (overwrite the original file)
        with open(os.path.join(attribute_director_path, attr_file_name), "w") as f:
            f.writelines(new_lines)

        altitude -= 12