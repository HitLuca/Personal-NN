# BackPropNeuralNetwork

## Introduction
This is a simple implementation of a feedforward neural network, created during spare time, and without the presumption to reach the accuracies of the well known libraries built especially for this purpose.  

The project is called BackPropNNv2, and as you may think it is a successor of my past BackPropNN. Now it has been deleted as this is far superior both in accuracy and efficiency (with BackPropNN i had to use a friend's pc to work with the full Mnist dataset).

For now it has to be used from the command line, until i decide to create a decent UI.

## Code Description

This project has been divided into various modules to be as easy to understand and use as possible

- [DatasetLoaders](https://github.com/HitLuca/BackPropNeuralNetwork/tree/master/src/DatasetLoaders)
    In this package I grouped all the dataset loaders that I have used until now. For now there is support only for the Mnist dataset and the iris one, but i plan to add more in the future.  
    If you want to add your loader i have created a java interface so you have there all the methods you need to implement.
    
- [Miscellaneous](https://github.com/HitLuca/BackPropNeuralNetwork/tree/master/src/Miscellaneous)
    Here you can find all the misc. functions that are supporting the main package, like number normalizing and image creation. As they don't are really part of the neural network I decided to move them here.  
    You will find also the normalizeMatrix function, that is especially useful when doing the backPropagation phase of the algorithm (use the nn in reverse to get the images).

- [NeuralNetwork](https://github.com/HitLuca/BackPropNeuralNetwork/tree/master/src/NeuralNetwork)
    Here lies the kernel of all the project, and is composed of two more packages and the NeuralNetwork class.
    
    - [Activations](https://github.com/HitLuca/BackPropNeuralNetwork/tree/master/src/Activations)
        I put all the activation functions that i used here, so you can add your favourite whenever you want.  As for now there are only two functions, the Sigmoid and the Logit(it's inverse).  
        I am planning to add the Softmax and other two or three when i have some time.
        
    - [CostFunctions](https://github.com/HitLuca/BackPropNeuralNetwork/tree/master/src/CostFunctions)
        Here you can find all the cost functions I have implemented, so yuo can choose your favourite.  I fixed the CrossEntropy function as active, for it's ability to avoid neuron saturation for the most times.  
        There is also another interface so you can see what to implement on your own.
        
    - [NeuralNetwork.java] (https://github.com/HitLuca/BackPropNeuralNetwork/blob/master/src/NeuralNetwork/NeuralNetwork.java)
        The main class, it has all the needed methods in the most easily readable way possible, but in case it needs some retouches just let me know.
    
- [Main.java] (https://github.com/HitLuca/BackPropNeuralNetwork/blob/master/src/Main.java)
    The wrapper, here you input all the parameters and import the desired database. I actually hate this so I plan to modify it as soon as I can.
    
## Motivation
I am studying at the Trento university, third year, and as I plan to continue my studies into the machine learning subject, I wanted to create a simple program to better understand all the shades of it, and for now it seems to have worked well.

## Installation
After cloning the repository you need to setup some stuff, mainly the folders used (planning to add a makefile or installation script), and add one of the supported datasets.  
First, create all the missing folders until you get this configuration:

- main folder
    - data/
        - iris/ (if used)
        - mnist/ (if used)
        - adult/ (if used)
    - lib/ (here you will have to put the jblas library)
        - [jblas-1.2.3.jar] (http://jblas.org/) you have to download this as it's used in the project
    - output/
        - imgs/
    - src/
        - Modules ecc.
       
Now you have to add your dataset in the right folder.  
With the mnist dataset there are a little things todo, and I plan to remove this part doing it on the loader: instead of the official mnist dataset you have to download the [csv version] (http://pjreddie.com/projects/mnist-in-csv/) and do some work at it. You need to load it up and normalize all it's values between 0 and 255 (i know, sorry), then save it to mnist_train.csv and mnist_test.csv.  
After this you are good to go.

[UPDATE] I have now added the two mnist datasets on my Google Drive folder, so you don't have to do all the mess mentioned above. Simply download the files listed [here] (https://drive.google.com/folderview?id=0B3ln413PhHWDeGhmRXJHWGV3UkE&usp=sharing) and move them inside the mnist folder.

## API Reference
Currently I haven't written any API reference, as for now it's pretty easy to use. Will ad as needed.

## Tests
Currently no test is needed.

## Performance 
As I said earlier, this software doesn't want to replace the more complex libraries, but in it's own it reaches good accuracies:
  
- Mnist dataset: The main benchmark, up to 99.82% accuracy on the train data (0.0882791763178177) and 97.69% (0.380672956122433) on the test data. The parameters I used were: learning rate 0.5, regularization value 5, miniBatch size 10, hidden neurons 100. These values were based on the Michael A. Nielsen book, reference below.
- Iris dataset: I tried only once so it's a bit of a joke run. By the way the accuracy went up to ~95% on the test data.

## References 
I created this project while reading the wonderful [Michael A. Nielsen, "Neural Networks and Deep Learning", Determination Press, 2015] (http://neuralnetworksanddeeplearning.com/index.html),
accompained by various papers and online documentation.  
Aside from them, I also have read the machine learning chapters in [The LION Way: Learning plus Intelligent Optimization] (http://www.amazon.com/dp/1496034023), adopting some tecniques described there.

## Contributors
I am working alone on this project, will add contributors as needed. If you want to help or simply give me a hint, feel free to ask.  
Mentioning [Alessandro994] (https://github.com/Alessandro994), as he offered to do benchmarks to the past version of the project. It was so unoptimized that my dual core processor was refusing to crunch all the Mnist dataset -.-

## License
This software is under the [MITLicense] (https://opensource.org/licenses/MIT)
Feel free to modify and redistribute it, but please add a mention to the original code.

## Kitty
This is a necessary addition

![alt text](http://3.bp.blogspot.com/-zMhJsRHNkX4/T9EffvGPDQI/AAAAAAAAC2A/dIoHZ8rHO14/s400/cat-yawn-gif.gif "Kitty is so sleepy he won't go away from the README.md file")
