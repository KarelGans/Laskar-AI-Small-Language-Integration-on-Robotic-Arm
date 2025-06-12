# Yolov8 Training for Robotic Arm Object Detection

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.
The dataset from this program can be obtained in a repository in https://drive.google.com/file/d/1ZCX3TcqLYUhyRF3E9J4XlpcDvAzgfxai/view?usp=sharing

### Prerequisites

-   Python 3.8 or later
-   pip package manager

### Installation & Usage

1.  **Install dependencies**
    This command will install all the necessary Python libraries listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the model**
    Run the training script to train your model on the dataset.
    ```bash
    python train_model.py
    ```

3.  **Run inference**
    Once the model is trained, use the inference script to make predictions on new data.
    ```bash
    python inference.py
