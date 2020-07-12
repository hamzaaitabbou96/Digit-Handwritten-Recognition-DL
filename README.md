  # Digit-Handwritten-Recognition DL

This project is focused on segmenting digits from images and their classification using deep learning.

Network's architure: CONV -> ReLU -> max pooling -> CONV -> RELU -> max pooling -> dropout -> flatten -> dense -> ReLU

Model is being trained on MNIST database, and then makes predictions on segmented images. Image has to contain digits only in one row - if it has mutliple rows This software is made for educational purposes and it's accuracy is quite poor for sample data. It's doing way better for the data produced manually (drawn in Paint for example).
