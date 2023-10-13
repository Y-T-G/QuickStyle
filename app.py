from pathlib import Path

from src.quick_paint_app import QuickPaintApp
from utils.utils import download_ir_model


if __name__ == "__main__":
    base_model_dir = "model"

    # Download Instance Segmentation model
    model_name = "instance-segmentation-security-1040"

    model_path = Path(f"{base_model_dir}/{model_name}.xml")
    if not model_path.exists():
        model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/temp/instance-segmentation-security-1040/FP16/instance-segmentation-security-1040.xml"
        download_ir_model(model_xml_url, base_model_dir)

    app = QuickPaintApp(
        "model/instance-segmentation-security-1040.xml",
        "model/gmcnn-places2-tf.xml",
    )

    app.launch()
