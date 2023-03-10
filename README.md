# Computer-Vision-ResNet-Object-Detector
Residual Neural Network Object Detector written for Pycocotool's library. Model implements custom skip block connections and uses a custom dataset loader for image classification object detection. This repository consists of 3 files, MSE, IOU and Val respectively. The first two, MSE and IoU are models that use different loss metrics for backpropagation; with MSE using Mean Square Error for back propagation and IoU using intersection over union. Running these two models will produce two separate neural nets, which can be further trained. The Val file takes these two models (saved as mse_net and iou_net) and evaluates its performance through data visualization of confusion matrices and producing randomly sampled images and indicates their predicted class labels and true labels. Note that the code requires two files, an annotations file containing the .json annotations for the data

This neural network architecture also utilizes bounding boxes, which are used to identify the dominant object of the image and make a prediction based on that dominant object. This can be seen throughout the code under the variable "bbox".

#Resnet Architecture
The architecture of the Resnet consists of several layers, including:

Convolutional layers: These layers apply a convolution operation to the input data, which involves sliding a set of filters over the data and computing dot products between the filter values and the input values. The output of a convolutional layer is a set of feature maps that capture different aspects of the input data.

Batch normalization layers: These layers normalize the output of the previous layer to have zero mean and unit variance, which can help improve the stability and speed of training.

ReLU activation layers: These layers apply the Rectified Linear Unit (ReLU) activation function to the output of the previous layer, which introduces non-linearity into the network.

Max pooling layers: These layers downsample the output of the previous layer by taking the maximum value within a sliding window.

Residual connections: These connections allow the network to skip over some of the layers during training. Specifically, the input to a residual block is added to the output of the block, which allows the network to "shortcut" the gradients and avoid the vanishing gradient problem.

Overall, the ResNet architecture in the code is designed to learn representations of images that can be used for classification tasks. By stacking multiple layers with residual connections, the network can learn increasingly complex and abstract features that can distinguish between different types of images.

# Summary of the Code
This code defines a customized dataloader that loads and preprocesses image data from the COCO dataset for object detection, then trains a convolutional neural network to perform classification and bounding box regression.  The mydataset class is a subclass of torch.utils.data.DataLoader that takes in an argument args that specifies the dataset path, the COCO annotation file path, a list of classes to consider, and a transformation function. The mydataset class processes the data, creating a list of dictionaries that contains image paths, bounding box coordinates, and label indices for each image in the dataset. The Namespace class is used to pass arguments to mydataset.  The ResnetBlock class defines a residual block for the neural network, and the HW5Net class defines the neural network architecture. The training function trains the neural network using cross-entropy loss for classification and mean squared error loss for bounding box regression.  The code uses PyTorch, NumPy, and OpenCV libraries for image processing, and Matplotlib for visualization.

#Residual Neural Networks
Residual neural networks, or ResNets for short, are a type of deep neural network that were first introduced in 2015. They were designed to address the problem of vanishing gradients in very deep neural networks by introducing skip connections, which allow information to flow directly from one layer to another without passing through intermediate layers.

The basic building block of a ResNet is called a residual block. A residual block consists of two convolutional layers, each followed by batch normalization and a ReLU activation function. The input to the block is passed through the first convolutional layer, and then the output of that layer is passed through the second convolutional layer. The output of the second convolutional layer is then added to the original input to create the residual connection. The resulting sum is then passed through a final ReLU activation function.

The skip connection in a residual block allows the network to learn residual functions that simply add to the identity mapping. This can be beneficial in cases where the optimal mapping is close to an identity mapping, as it allows the network to learn the residual deviations from the identity. Additionally, the skip connection allows gradients to flow directly from later layers to earlier layers, helping to address the vanishing gradient problem.

In practice, ResNets typically consist of many stacked residual blocks, with downsampling performed using strided convolutional layers or max pooling. The output of the final convolutional layer is typically passed through one or more fully connected layers to produce the final output.

Overall, the key innovation of ResNets is the introduction of skip connections, which allow information to flow directly from one layer to another, helping to address the vanishing gradient problem and enabling the training of very deep neural networks.
