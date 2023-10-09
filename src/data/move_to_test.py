import os
import shutil
import random


def move_images_to_test(root_folder: str, percentage: float):
    """
    Mueve un porcentaje especificado de imágenes desde una carpeta raíz a una carpeta de prueba.

    Args:
        root_folder (str): La ruta a la carpeta raíz que contiene las subcarpetas con imágenes.
        percentage (float): El porcentaje de imágenes a mover a la carpeta de prueba.

    Returns:
        None
    """
    test_folder = os.path.join(os.path.dirname(root_folder), "test")
    os.makedirs(test_folder, exist_ok=True)

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            test_subfolder = os.path.join(test_folder, folder_name)
            os.makedirs(test_subfolder, exist_ok=True)

            image_files = [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp"))
            ]

            num_images_to_move = int(len(image_files) * percentage)

            images_to_move = random.sample(image_files, num_images_to_move)

            for image in images_to_move:
                src_path = os.path.join(folder_path, image)
                dest_path = os.path.join(test_subfolder, image)
                shutil.move(src_path, dest_path)
                print(f"{src_path} to {dest_path}")


root_folder = "data/train"
percentage = 0.25
move_images_to_test(root_folder, percentage)
