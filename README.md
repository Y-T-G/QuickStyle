# QuickStyle

A simple stylizing app utilizing OpenVINO to stylize common objects in images.

It uses [instance-segmentation-security-1040](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/instance-segmentation-security-1040#instance-segmentation-security-1040) model from Open Model Zoo for instance segementation and [Fast Neural Style Transfer](https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model) for stylizing.

![Demo](assets/demo.png)

## Usage

1. Install dependencies

    ````bash
    pip3 install -r requirements.txt
    ````

2. Run the app

    ````bash
    python3 app.py
    ````
