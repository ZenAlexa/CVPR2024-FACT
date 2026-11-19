import os

def get_project_base():
    # 直接返回当前目录作为项目基础路径
    return os.path.dirname(os.path.realpath(__file__)) + "/"


if __name__ == "__main__":
    print(get_project_base())
