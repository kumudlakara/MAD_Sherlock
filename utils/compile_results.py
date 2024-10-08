import json

def add_result(file_path, annotation):
    try:
        # Read the existing file
        with open(file_path, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        # If the file doesn't exist or is empty, initialize an empty list
        data = []

    # Append the new result to the list
    data.append(annotation)

    # Write the updated list to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def add_result_with_disambiguation(file_path, annotation, search_results, queries):
    try:
        # Read the existing file
        with open(file_path, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        # If the file doesn't exist or is empty, initialize an empty list
        data = []

    # Append the new result to the list
    data.append(annotation)
    data.append(search_results)
    data.append(queries)

    # Write the updated list to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
