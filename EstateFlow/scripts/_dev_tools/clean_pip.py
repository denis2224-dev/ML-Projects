import pkg_resources
import os

def update_requirements():

    target_path = os.path.join(os.path.dirname(__file__), "..", "..", "requirements.txt")
    target_path = os.path.normpath(target_path)

    packages = sorted([f"{d.project_name}=={d.version}" for d in pkg_resources.working_set])

    with open(target_path, "w") as f:
        for package in packages:
            f.write(package + "\n")

    print(f"EstateFlow: requirements.txt has been updated!")


if __name__ == "__main__":
    update_requirements()