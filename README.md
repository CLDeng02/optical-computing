# optical-computing
As a complex-valued neural network, the optical diffractive neural network functions in pattern recognition and Boolean logic operations despite lacking the nonlinear activation function component.<br>  

The code corresponds to the model in the figure below. Training the network via the backpropagation algorithm enables the establishment of an end-to-end optical computing model that maps inputs to outputs.This figure is from the paper [Diffraction casting](https://www.spiedigitallibrary.org/journals/advanced-photonics/volume-6/issue-5/056005/Diffraction-casting/10.1117/1.AP.6.5.056005.full?tab=ArticleLinkCited) <br>
| :-------: | :-------: |
|！<img width="333" height="188" alt="image" src="https://github.com/user-attachments/assets/113cdc28-950e-480a-b2f0-b7d811bcad3e" /> | ！
<img width="333" height="188" alt="image" src="https://github.com/user-attachments/assets/df28f319-c052-4ae3-a86f-2b605b5649a5" /> | <br>
The figure below shows the error reduction curve and simulation results obtained from the training set.<br>
<img width="333" height="188" alt="image" src="https://github.com/user-attachments/assets/091389b8-a2f8-4928-bba5-35cff6510e28" /> <br>
<img width="333" height="188" alt="image" src="https://github.com/user-attachments/assets/57f5760f-7bc3-46bd-8939-ff0e8e52c75a" /> <br>
Since the original code has not been made public, the code for this project is written by myself. For a detailed explanation of the relevant principles, please refer to the file: **Diffractive optical computing.pdf**. <br>
Currently, the simulation multiplexing of 100 pairs of images can be achieved in the training set. Through test comparison, using the gradient descent algorithm, its performance is far superior to the wavefront matching algorithm and the phase retrieval iterative algorithm. However, in the test set, the transfer generalization effect is very poor.It is currently unclear whether the original author has any undisclosed details.
