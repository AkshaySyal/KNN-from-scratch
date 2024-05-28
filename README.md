# KNN-from-scratch
Implemented KNN from scratch using numpy on MNIST dataset (60k training, 10k test)

Given number of training samples is n each having d number of features

## Time Complexity: 

Training: KNN has no explicit training. Hence O(1)

Prediction: Θ(n*d) 

Given a test sample its distance is calculated against n training samplex. Computing Euclidean distance takes Θ(d).
Store only k shortest distances in an array. After appending distance sort the array. Pop out last element if length of array exceeds k.

Space Complexity: Θ(n*d) 
Have to store the entire training data at run time during prediction
