import os
from typing import Any
from uuid import uuid4

from utils import join_paths


class ImageSaver:
    @staticmethod
    def save_images(
        output_path: str,
        images: Any,
        folder_name: str = "",
        format: str = ".png",
    ) -> None:
        for image in images:
            image_id = uuid4()
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            if folder_name:
                out_path = join_paths(
                    output_path,
                    folder_name,
                )
            else:
                out_path = output_path

            if not os.path.exists(out_path):
                os.mkdir(out_path)

            image.save(join_paths(out_path, f"{image_id}.{format}"))
