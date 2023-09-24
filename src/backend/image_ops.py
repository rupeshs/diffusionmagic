from PIL import Image, ImageEnhance


def resize_pil_image(
    pil_image: Image,
    image_width,
    image_height,
):
    return pil_image.convert("RGB").resize(
        (
            image_width,
            image_height,
        ),
        Image.Resampling.LANCZOS,
    )


def get_black_and_white_image(pil_image: Image):
    img_enhance = ImageEnhance.Color(pil_image)
    bw_image = img_enhance.enhance(0)
    return bw_image
