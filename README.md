# ComfyUI-Animator

This custom node for ComfyUI provides full-body animation capabilities, including facial rigging, various lighting styles, and green screen output.

## Features

- Full body detection and rigging
- Facial rigging for detailed animation
- Animation based on input driving video
- Multiple lighting styles
- Green screen output option

## Installation

1. Clone this repository into your ComfyUI custom nodes folder:
   ```
   git clone https://github.com/SEkINVR/ComfyUI-Animator.git
   ```
   
2. Install the required dependencies:
   ```
   cd ComfyUI-Animator./install.sh
   ```
3.## Usage

1. In ComfyUI, you'll find the "Full Body Animator" node in the "animation" category.
2. Connect an image input, a driving video input, and select a lighting style.
3. The node will output two videos: one with the animated and lit subject, and another with the subject on a green screen background.

## Updating

To update the node to the latest version, run:
```
./update.sh
```

## License

[Your chosen license]
4.requirements.txt:
```
torch>=1.7.0
numpy>=1.19.0
Pillow>=8.0.0
opencv-python>=4.5.0
mediapipe>=0.8.9
```
5.install.sh:
```
#!/bin/bash
pip install -r requirements.txt
```
6. update.sh:
```
#!/bin/bash
git pull
pip install -r requirements.txt
```
Here are some important points:

We're using MediaPipe for body pose estimation and facial landmark detection.
The animation logic (animate method) is a placeholder. In a real implementation, you'd need to use a more sophisticated method like First Order Motion Model.
The lighting application (apply_lighting method) is simplified. Real lighting would involve more complex 3D rendering techniques.
For background removal, we're using a pre-trained DeepLabV3 model for semantic segmentation.

To make this work with ComfyUI, you'll need to ensure that the input and output formats are compatible. ComfyUI typically works with PyTorch tensors, so you might need to convert between PIL Images, numpy arrays, and PyTorch tensors at various points in your pipeline.
Also, note that this implementation assumes that driving_video is a list of frames. You might need to adjust this depending on how ComfyUI handles video input.
Lastly, remember to add error handling, input validation, and performance optimizations. This implementation is a starting point and would need to be refined and tested thoroughly for production use.
