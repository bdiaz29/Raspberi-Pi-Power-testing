# Raspberi-Pi-Power-testing
scripts to run to test power consumeption of raspberi pi for diffrent neural network architectures.

The scripts are to run either a mobilenet architecture or a resnet architecture with each running either as a 32 bit
floating point model or a 16 bit floating point model, two things will be measured power consumption and timing. 
each architecure is modified to take in 112 by 112 and draw a bound around a face within the picture. 
both h5 and a 16 bit floating point quantized tflite models are used, for both models and the 32 and 16 bit models different sized
batches are tested ranging from 1 to 32, at each seaction the power of the device is mesured externally to determine the 
diffrent power draw for diffrent architecutes. The timing of each architecure is measured in software. 
