# Image-Classifier

Large-scale Image Recognition using a variation of VGG NET.

1.	INSTRUCTIONS FOR RUNNING CODE:

 •	 “train.py” has the entire code that classifies image, evaluates and plots output.

 •	 “cifar10.py” which is used to load and extract parameters from the cifar-10 dataset from the directory where the dataset is downloaded and stored.

 •	Specify the directory where dataset is downloaded in line 25 of “train.py”. cifar10.data_path = "<Directory>"

 •	Execution command.  python train.py --lr 0.01 --batch_size 20 --init 1 --save_dir "<directory to store model and checkpoints>"

2. THE OVERALL STRUCTURE OF THE NETWORK: 

 (a) CONV1: convolutional layer with 3 inputs (RGB), 64 outputs (ﬁlter size is thus 64 × 3 × 3 × 3) 

 (b) POOL1: 2 × 2 max-pooling layer 

 (c) CONV2: convolutional layer with 64 outputs, 128 outputs (ﬁlter size 128 × 64 × 3 × 3) 

 (d) POOL2: 2 × 2 max-pooling layer 

 (e) CONV3: convolutional layer with 128 inputs, 256 outputs 

 (f) CONV4: convolutional layer with 256 inputs, 256 outputs 

 (g) POOL3: 2 × 2 max-pooling layer 

 (h) FC1: fully connected layer with 256 inputs, 1024 outputs 

 (i) FC2: fully connected layer with 1024 inputs, 1024 outputs 

 (j) SOFTMAX: softmax layer for classiﬁcation: 1024 inputs, 10 outputs

3. ADDITIONAL INFORMATION

 • For all convolution layers, stride S = 1, padding P = 1 

 • All layers, except for the pooling layers and for the last (softmax-)layer should use ReLU-nonlinearities. 

 • Network is trained using Adam with momentum using 45000 randomly sampled examples from the training dataset. The remaining 5000 examples is used for validation. 

 • Use batch-normalization on the last layer activations (immediately before computing the softmax) when training the network. 

 • Diﬀerent parameter initializations like Xavier, He etc. is used to ﬁnd settings that are stable and yield good solutions. 

 • Used early stopping using the validation set with a patience of 5 epochs. 

 • Code requires following inputs: 
  
  – --lr (initial learning rate η for gradient descent based algorithms)
  
  – --batch size (the batch size to be used - valid values are 1 and multiples of 5) 
  
  – --init (the initialization method to be used - 1 for Xavier, 2 for He) 
  
  – --save dir (the directory in which the pickled model should be saved - by model we mean all the weights and biases of the network)
