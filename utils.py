import json


def debug_wrapper(func):
    def inner_func(*args, **kwargs):
        print("********************** BEGIN **********************")
        func(*args, **kwargs)
        print("********************** END **********************")

    return inner_func


def write_to_file(obj, file_path):
    with file_path.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=4)
    print(f'objects is written to {file_path}')
