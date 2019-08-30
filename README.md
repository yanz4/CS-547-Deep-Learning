# CS-547-Deep-Learning



<BODY BGCOLOR="White" LINK="blue" VLINK="blue" ALINK="blue">



<TABLE WIDTH="800" BORDER="0" CELLSPACING="0" CELLPADDING="2" ALIGN="LEFT" VALIGN="TOP">





<P><FONT SIZE="6" COLOR=#000000><b> <font color="black"> Deep Learning </font>      </b></FONT></P>


<p> CS 547/ IE 534, Fall 2019 <p>
<p> Instructor: Justin Sirignano<br>

Teaching Assistant: Yuanyi Zhong, Xiaobo Dong, Lei Fan, Rachneet Kaur, Jyoti Aneja, Peijun Xiao <br> 
<br />
</p>


</TABLE>

<TABLE WIDTH="900" BORDER="0" CELLSPACING="0" CELLPADDING="0" ALIGN="LEFT" VALIGN="TOP">
<TR><TD WIDTH="900" ALIGN="LEFT" VALIGN="TOP">




<FONT SIZE="4" COLOR = #000000><b>Course overview</b></FONT><p>

<p> Topics include convolution neural networks, recurrent neural networks, and deep reinforcement learning. Homeworks on image classification, video recognition, and deep reinforcement learning. Training of deep learning models using TensorFlow and PyTorch. A large amount of GPU resources are provided to the class. See <A HREF="https://courses.engr.illinois.edu/ie534/fa2019/Deep Learning Fall 2019 Syllabus.pdf" target="_blank"><b>Syllabus</B></A> for more details.


<p> Mathematical analysis of neural networks, reinforcement learning, and stochastic gradient descent algorithms will also be covered in lectures. <p>

<p> IE 534 Deep Learning is cross-listed with CS 547. <p>

<p> This course is part of the Deep Learning sequence: <p>
<ul>  
	<li> IE 398 Deep Learning (undergraduate version) </li>  
	<li> IE 534 Deep Learning  </li> 
	<li> IE 598 Deep Learning II   </li> 
</ul>


<FONT SIZE="4" COLOR = #000000><b>Computational resources </b></FONT><p>

<p> A large amount of GPU resources are provided to the class: 100,000 hours. Graphics processing units (GPUs) can massively parallelize the training of deep learning models.  This is a unique opportunity for students to develop sophisticated deep learning models at large scales. <p>

<FONT SIZE="4" COLOR = #000000><b> Code </b></FONT><p> 
<p> Extensive TensorFlow and PyTorch code is provided to students. <p>

<FONT SIZE="4" COLOR = #000000><b> Datasets, Code, and Notes </b></FONT><p> 
<p> <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/MNISTdata.hdf5" target="_blank"><b> MNIST Dataset </B></A> <p>
<p> <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/CIFAR10.hdf5" target="_blank"><b> CIFAR10 Dataset </B></A> <p>
<p> <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/Blue_Waters_Introduction.pdf" target="_blank"><b> Introduction to running jobs on Blue Waters </B></A> <p>
<p> <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/Blue_Waters_Help_Document.pdf" target="_blank"><b> Blue Waters Help Document for the Class </B></A> <p>
<p> <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/Deep_Learning_Papers.pdf" target="_blank"><b> Recommended articles on deep learning </B></A> <p>
<p> <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/Pytorch_Tutorial.pdf" target="_blank"><b> PyTorch Class Tutorial </B></A> <p>
<p> <A HREF="https://pytorch.org/tutorials/" target="_blank"><b> PyTorch Website </B></A> <p>
<p> <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/Deep Learning Lecture Notes UIUC.pdf" target="_blank"><b> Course Notes for Weeks 1 & 2 </B></A> <p>



<p> Lecture Slides: <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/Lecture 1.pdf" target="_blank"><b> Lecture 1 </B></A>, <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/Lecture 2.pdf" target="_blank"><b> Lecture 2-3 </B></A>, <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/Lecture 4.pdf" target="_blank"><b> Lecture 4-5 </B></A>, <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/Lecture 6.pdf" target="_blank"><b> Lecture 6 </B></A>, <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/Lecture 8.pdf" target="_blank"><b> Lecture 8 </B></A>, <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/Lecture 10.pdf" target="_blank"><b> Lecture 10 </B></A>, <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/GAN_presentation.pdf" target="_blank"><b> GAN Lecture Slides </B></A>, <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/Lecture 11.pdf" target="_blank"><b> Lecture 11 </B></A>, <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/PyTorch_Distr.py" target="_blank"><b> Code for Distributed Training </B></A>, <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/Lecture 12.pdf" target="_blank"><b> Lecture 12 </B></A> , <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/Deep Learning Image Ranking Presentation.pdf" target="_blank"><b> Deep Learning Image Ranking Lecture </B></A>, <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/secure/AR_presentation.pdf" target="_blank"><b> Action Recognition Lecture </B></A>  <p>



<FONT SIZE="4" COLOR = #000000><b> Homeworks </b></FONT>



<ul> 
HW1: Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch). The neural network should be trained on the Training Set using stochastic gradient descent. It should achieve 97-98% accuracy on the Test Set. For full credit, submit via Compass (1) the code and (2) a paragraph (in a PDF document) which states the Test Accuracy and briefly describes the implementation. Due September 6 at 5:00 PM.  

HW2: Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch). You should write your own code for convolutions (e.g., do not use SciPy's convolution function). The convolution network should have a single hidden layer with multiple channels. It should achieve at least 94% accuracy on the Test Set.  For full credit, submit via Compass (1) the code and (2) a paragraph (in a PDF document) which states the Test Accuracy and briefly describes the implementation. Due September 13 at 5:00 PM.  

HW3: Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset. The convolution network should use (A) dropout, (B) trained with RMSprop or ADAM, and (C) data augmentation. For 10% extra credit, compare dropout test accuracy (i) using the heuristic prediction rule and (ii) Monte Carlo simulation. For full credit, the model should achieve 80-90% Test Accuracy. Submit via Compass (1) the code and (2) a paragraph (in a PDF document) which reports the results and briefly describes the model architecture. Due September 27 at 5:00 PM.  

HW4: Implement a deep residual neural network for CIFAR100. <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/IE534_HW4.pdf" target="_blank"><b> Homework #4 Details. </B></A>  Due October 11 at 5:00 PM. 

HW5: Generative adversarial networks (GANs).  <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/GAN.html" target="_blank"><b> Homework Link </B></A> Due 5 PM, October 18.

HW6: Natural Language Processing A. Due October 25 at 5 PM. <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/NLP.html" target="_blank"><b> Part I and II of NLP assignment </B></A> <p> 

HW7: Natural Language Processing B. Due November 1. <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/NLP.html" target="_blank"><b> Part III of NLP assignment </B></A> <p> 

HW8: Video recognition I. Due November 15.  <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/AR.html" target="_blank"><b>Homework Link</B></A>

HW9: Deep reinforcement learning on Atari games. Due November 29.  <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/AR.html" target="_blank"><b>Homework Link</B></A>

HW10 (not assigned this year): Implement a deep learning model for image ranking. <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/ImageRankingProject.pdf" target="_blank"><b> Homework #5 Details. </B></A>  Due October 18 at 5:00 PM. 

HW11 (not assigned this year): Deep reinforcement learning on Atari games I using TensorFlow.  <A HREF="https://github.com/tgangwani/IE598_RL/tree/master/hw6" target="_blank"><b>2017 version of this homework</B></A>. 
HW12 (not assigned this year): Deep reinforcement learning on Atari games II using TensorFlow.  <A HREF="https://github.com/tgangwani/IE598_RL/tree/master/hw7" target="_blank"><b>2017 version of this homework</B></A>.


Final Project: See Syllabus for a list of possible final projects. Due December 12. Examples of Final Projects: <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/siqim2.pdf" target="_blank"><b> Image Captioning I </B></A>, <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/Project_report.pdf" target="_blank"><b> Faster RCNN </B></A>, <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/Final_report.pdf" target="_blank"><b> Image Captioning II </B></A>,  <A HREF="https://courses.engr.illinois.edu/ie534/fa2018/report_BootstrappedDQN_qchen35_hli70_mlihan2.pdf" target="_blank"><b> Deep Reinforcement Learning </B></A>.


</ul>




    



</TABLE>





</BODY>

</HTML>
