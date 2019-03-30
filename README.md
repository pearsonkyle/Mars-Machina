# Mars-Machina
A deep convolutional variational autoencoder trained on digital terrain maps of Mars from HiRise/MRO. 3D surfaces can be procedurally generated from the latent space

![](https://github.com/pearsonkyle/Mars-Machina/blob/master/Mar%2029%202019%205_27%20PM%20-%20Edited.gif)

## Dependencies
- Unity 3D
- [Unity ML Agents](https://github.com/llSourcell/Unity_ML_Agents/tree/master/docs)
- https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/Using-TensorFlow-Sharp-in-Unity-(Experimental).md
- Python 3.6+
- Keras, Tensorflow, Matplotlib, Numpy, PIL, Scikit-learn

## Python
A [DCVAE](https://github.com/chaitanya100100/VAE-for-Image-Generation) is trained on 1000 random sections from all images in Python/hirise/*.png 

1. Download a [Digital terrain map](https://www.uahirise.org/dtm/) from HiRise
![](https://github.com/pearsonkyle/Mars-Machina/blob/master/hirise_web.png)

Save to Python/hirise/ as a png file. Make sure to trim the unnecessary regions in GIMP or photoshop before training. See the current file for an example

Train a quick model: 
```
python autoencoder.py --lose mse --epochs 1000 --name hirise
```
a tensorflow graph will be saved to: Python/out/frozen_hirise.bytes

Create a game object in unity and give it some mesh components, then attach the meshGenerator.cs script

![](https://github.com/pearsonkyle/Mars-Machina/blob/master/unity_loadmodel.png)

For getting started with Tensorflow in Unity see: https://github.com/pearsonkyle/Unity-Variational-Autoencoder
