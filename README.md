# Machine_Learning_Tool_4
This is a single-filed machine learning platform. All the layers that are required for computer vision are included in just a one python file. 
It utilizes Cupy instead of numpy for fast computing. 
If you already can use keras, you probably can learn to use this tool in no time.

Features:

 Layers:
 
 -Dense layer
 
 -Convolutional 2D layer
 
 -Average pooling 2D layer
 
 -Flatten layer
 
 Loss functions:
 
 -MSE
 
 -Categorical Crossentropy(with bugs, which should be fixed soon)
 
 Loading and Saving:
 
 -Load images and labels (optional)
 
 -Easy weight saving / loading (optional)
 
 Else:
 -Easy result prediction
 
 -Able to interect with running code*(explain in bottom) (optional)
 
 -Not_static_learning_rate: learning rate changes every epoch to minimize the cost faster. (optional)
 
 -Check accuracy with one line of code. (optional)
 
Bugs to fix:

 -model doesn't learn when using categorical crossentropy loss function.
 
*About Running Code Interection:  

 With other machine learning platforms, you can't change configurations of the model while the program is running. But with this tool, you can. 
 When you create model class, type in the directory of txt file which includes the code that you want to execute at the beginning of every epochs. 
 You can change that code while training is on going. 
 With this feature, you can do something like this:
 
 -Pause training
 
 -Release the pause
 
 -Save model's weights while training
 
 -Reload model's weights while training
 
 -Change model's learning rate
 
 -Change optimizer (gd to sgd)
 
 -Change Loss function
 
 -Switch between static learning rate and not static learning rate
 
 You can do basically anything that you want if it is written in python.
