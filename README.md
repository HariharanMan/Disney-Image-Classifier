# Disney Character Classification using Streamlit and CNN

## Overview

This project is a web application built using Streamlit that classifies images of Disney characters as either *Olaf* or *Pumba*. It uses a Convolutional Neural Network (CNN) model trained on the Disney Characters Dataset, leveraging TensorFlow and Keras for deep learning.

The application includes:

- Dataset downloading and cleaning from Kaggle.
- Training a CNN model with progress visualization.
- Uploading an image for classification through an intuitive interface.

---

## Features

1. **Dataset Management**:
   - Automatically downloads the Disney Characters dataset from Kaggle.
   - Cleans corrupted images to ensure smooth model training.

2. **Model Training**:
   - Trains a CNN with live progress bars for a better user experience.

3. **Image Classification**:
   - Accepts image uploads (JPEG, JPG, PNG formats) and predicts whether the character is Olaf or Pumba.
   - Displays results in real time with intuitive UI enhancements.

---

## Installation

### Prerequisites

- Python 3.8 or above.
- Install required Python libraries using the following command:

```bash
pip install -r requirements.txt
