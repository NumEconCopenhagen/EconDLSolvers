import os
import datetime
import json

def clean_solving_json():

    filename = "solving.json"
    with open(filename, 'w') as file:
        json.dump([], file)

def solving_json(postfix):

    filename = "solving.json"

    # check if the file exists
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            json.dump([], file)

    # read the existing data from the file
    with open(filename, 'r') as file:
        data = json.load(file)

    # add a new row with the current timestamp and the value false
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {}
    if not postfix == '': new_entry['postfix'] = postfix
    new_entry['timestamp'] = timestamp
    new_entry['terminate'] = False
    data.append(new_entry)

    # write the updated data back to the file
    with open(filename, 'w') as file:
        json.dump(data,file,indent=4)

    return timestamp

def check_solving_json(timestamp):

    filename = "solving.json"

    try:

        with open(filename, 'r') as file:
            data = json.load(file)

        for entry in data:
            if entry.get('timestamp') == timestamp:
                terminate = entry.get('terminate', None)

        return terminate == True

    except Exception:

        return False