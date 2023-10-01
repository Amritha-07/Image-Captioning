# Image-Captioning

A Web Application for Generating Captions for Images

This project carries out Image Captioning with the help of hugging face model and implemented it in streamlit

## Tech-Stack

- Python

## Features
- The application allows the user to upload an image in JPG or PNG format.
- The application uses the nlpconnect/vit-gpt2-image-captioning model from hugging face, which is a combination of Vision Transformer (ViT) and GPT-2, to generate captions for images.
- The application displays the caption generated by the model.

## Installation and Usage

To install and run this project, you need to have Python 3 installed on your computer. You also need to install and import the following Python libraries:

streamlit
transformers
PIL
requests

You can use any IDE or editor of your choice.

To download the code for this project, you can clone this GitHub repository using the following command:

```git clone https://github.com/Amritha-07/Image-Captioning.git```

To execute the code, you can run the following command:

```streamlit run app.py```

## References

- [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) from Hugging Face
