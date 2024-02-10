# QuickStyle

![Mentioned in Awesome OpenVINO](https://camo.githubusercontent.com/9dbe56f475e0abb1d07a9a18d841378c3d40325ae7df300f29491158cf38d013/68747470733a2f2f617765736f6d652e72652f6d656e74696f6e65642d62616467652d666c61742e737667)

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
