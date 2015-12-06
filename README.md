# BackPropNeuralNetwork

## Introduction
This is a simple implementation of a feedforward neural network, created during spare time, and without the presumption to reach the accuracies of the well known libraries built especially for this purpose.

Tje project is called BackPropNNv2, and as you may think it is a successor of my past BackPropNN. Now it has been deleted as this is far superior both in accuracy and efficiency (with BackPropNN i had to use a friend's pc to work with the full Mnist dataset).

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
        I put all the activation functions that i used here, so you can add your favourite whenever you want. As for now there are only two functions, the Sigmoid and the Logit(it's inverse).
        I am planning to add the Softmax and other two or three when i have some time.
        
    - [CostFunctions](https://github.com/HitLuca/BackPropNeuralNetwork/tree/master/src/CostFunctions)
        Here you can find all the cost functions I have implemented, so yuo can choose your favourite. I fixed the CrossEntropy function as active, for it's ability to avoid neuron saturation for the most times.
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
        - jblas-1.2.3.jar YOU HAVE TO DOWNLOAD THIS OR IT WILL NOT WORK
    - output/
        - imgs/
    - src/
        - Modules ecc.
        
## API Reference
Currently I haven't written any API reference, as for now it's pretty easy to use. Will ad as needed

## Tests
Currently no test is needed.

## References 
I created this project while reading the wonderful [Michael A. Nielsen, "Neural Networks and Deep Learning", Determination Press, 2015] (http://neuralnetworksanddeeplearning.com/index.html),
accompained by various papers and online documentation.
Aside from them, I also have read the machine learning chapters in [The LION Way: Learning plus Intelligent Optimization] (http://www.amazon.com/dp/1496034023), adopting some tecniques described there.

## Contributors
I am working alone on this project, will add contributors as needed. If you want to help or simply give me a hint, feel free to ask.

## License
This software is under the [MITLicense] (https://opensource.org/licenses/MIT)
Feel free to modify and redistribute it, but please add a mention to the original code.