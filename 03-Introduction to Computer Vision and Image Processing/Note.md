# Week 1

## INTRODUCTION TO COMPUTER VISION

1. **Introduction to Computer Vision**: The course is an introduction to computer vision, focusing on understanding its applications and implementing techniques using Python and OpenCV.

2. **Course Objectives**: By the end of the course, students will:
   - Understand what computer vision is.
   - Apply computer vision algorithms with Python and OpenCV.
   - Create custom classifiers for practical use.
   - Build a web app for image classification.

3. **What is Computer Vision?**: Computer vision is the field that gives computers the ability to see and understand images, similar to how humans interpret visual data.

4. **Impact on Industries**: Computer vision has disrupted various industries by:
   - Increasing efficiency and automation.
   - Reducing costs.
   - Enabling scalability.
   - Improving safety.

5. **Case Studies**:
   - **ADNOC**: Used computer vision to classify rock samples, saving time and effort for geologists.
   - **Knockri**: Employed AI video assessments for soft skill evaluations in HR, streamlining the hiring process.

The script provides examples of how computer vision technologies are making a significant impact across different sectors.

## Applications of Computer Vision
1. **Video Searchability**: IBM has developed a system that tags videos with keywords based on the objects appearing in each scene. This enables users to search for specific scenes or content within videos, making video navigation more efficient.

2. **Security Footage Analysis**: Computer vision and object recognition are transforming the way security companies operate. Instead of manually sifting through hours of video footage to find suspects or specific objects (like a blue van), computer vision can automate this process, significantly improving efficiency.

3. **Infrastructure Maintenance**: Industries like civil engineering face challenges such as maintaining structures like electric towers. Climbing and inspecting these towers manually is time-consuming and risky. Computer vision can be applied here by using high-resolution images taken from different angles. These images are divided into smaller grids, and custom image classifiers are developed to detect the presence of metal structures, rust levels, and other structural defects. This approach not only enhances safety but also saves time and costs.

4. **Damage Assessment for Insurance**: Insurance companies can benefit from computer vision for assessing claims. Custom classifiers can be created to classify damage severity, such as identifying different levels of rust or classifying damage caused by hail and storms. This automated classification can streamline the claims processing workflow, potentially saving time and money.

5. **Severity Grading**: Grading the severity of claims can be challenging for humans. Computer vision can provide an objective and consistent way to assess damage severity, making the insurance claim process more efficient.

The video showcases how computer vision techniques can be applied across various industries to improve efficiency, safety, and cost-effectiveness.

## Recent Research in Computer Vision
Discussing some active research areas in the field of computer vision over the past decade. Here are the key points:

1. **Object Detection**: Researchers at Facebook are actively working on detecting objects in images. Accurate and efficient object detection is crucial in computer vision, as it serves as the foundation for making meaningful inferences in images and video streams. This is particularly relevant in the context of self-driving cars, where cameras need to detect objects in real-time to ensure safe navigation and collision avoidance.

2. **Image-to-Image Translation**: Image-to-image translation is another exciting area of research. It involves transforming an image from one representation to another. For example, researchers are working on techniques to convert images of horses into zebras and vice versa. Similarly, they are exploring methods to change the season or weather conditions in an image, such as converting a summer scene to a winter scene.

3. **Motion Transfer**: The UC Berkeley Research Team is involved in projects like "Everybody Dance Now," which use computer vision techniques for motion transfer. In this context, if there's a video of a person performing dance moves, computer vision can be used to transfer those dance moves onto an amateur dancer or another target.

These research areas demonstrate the evolving capabilities of computer vision, ranging from practical applications like object detection in self-driving cars to creative endeavors like image-to-image translation and motion transfer.

## Brainstorming your Own Applications
In this video, the narrator encourages you to brainstorm ideas for computer vision applications. Here are the key takeaways:

1. **Start with Existing Problems**: Rather than starting with a solution in mind, begin by identifying existing problems that can be addressed using computer vision. For example, consider issues that people face in their daily lives, at work, or in various industries.

2. **Narrow Down by Industry**: To help focus your brainstorming, consider different industries where computer vision can make an impact. The narrator mentions several fields, including medicine, driving, security, surveillance, manufacturing, insurance, and work safety.

3. **Examples of Problem Areas**: The video provides examples of problem areas within these industries:
   - In medicine, training doctors to accurately detect cancer can be challenging.
   - Driving requires constant visual attention, and fatigue-related accidents are a concern.
   - The security and surveillance industry often involves sifting through hours of footage to find suspects.
   - Ensuring product quality in manufacturing can be labor-intensive.
   - Monitoring compliance with safety equipment in construction.
   - Streamlining the assessment of car accident damage for insurance claims.

4. **Apply to Your Life**: The narrator suggests thinking about problems you encounter personally, whether it's tracking receipts, identifying plants, dealing with pests, or any other daily challenges.

5. **Endless Possibilities**: Remember that there are endless possibilities for computer vision applications. You can make your life, your friends' lives, or your work more convenient and efficient with creative solutions.

The goal is to identify real-world problems and explore how computer vision technology can be used to solve them. Starting with the problem and then finding innovative ways to apply computer vision is a great approach to generating valuable ideas for future projects or applications.

# Week 2 Image Processing with OpenCV and Pillow

## What Is a Digital Image?

A digital image can be thought of as a rectangular array of numbers. In many cases, we work with grayscale images, which are composed of different shades of gray. If we zoom in on a region of such an image, we see that it's made up of a rectangular grid of blocks called pixels. Each pixel is associated with a number called an intensity value, which represents the pixel's shade of gray. Digital images typically use intensity values ranging from 0 to 255, providing 256 different shades of gray.

The following bar demonstrates the relationship between shades of gray and numerical values. Darker shades have lower values (0 being black), while lighter shades have higher values (255 being white). The contrast in an image is determined by the difference in intensity values.

Reducing the number of intensity values can affect image quality. For example, with only 32 intensity values, the image still looks similar to the original. However, as we reduce the values further, the image quality deteriorates, particularly in regions of low contrast.

## Image Representation

In Python, an image is represented as a rectangular array of numbers. The height corresponds to the number of rows, while the width corresponds to the number of columns. Each pixel or intensity value is indexed by its row and column position.

In color images, such as RGB images, each pixel is associated with multiple intensity values, one for each color channel (e.g., red, green, blue). These channels combine to create the color of the pixel. So, a color image is like a cube, with each channel having its own intensity values.

Additionally, black and white images can be represented as binary masks, where one intensity value represents the object of interest, and another represents the background.

Video sequences are a series of images, each representing a frame in the video.

## Working with Images in Python

To work with images in Python, you'll often need to load them using libraries such as Pillow (PIL) or OpenCV.

### Pillow (PIL)

- To use the Pillow library for image processing, you can import the image module from PIL.

- You can load an image using the `Image.open()` method and create a PIL image object.

- Displaying the image can be done with `show()` or by using `matplotlib`'s `imshow()`.

- The attributes `format`, `size`, and `mode` provide information about the image format, dimensions, and color mode.

- The `ImageOps` module provides various image processing operations, including converting to grayscale, saving images, and quantizing.

- Gray-scale images have a mode of "L" (luminance).

- You can split color channels using the `split()` method.

- To save an image, you can use the `save()` method and specify the format.

### OpenCV

- OpenCV is a library for computer vision with more functionality than Pillow but is more complex.

- You can import OpenCV as `cv2`.

- Use `cv2.imread()` to load an image. The result is a numpy array with intensity values.

- The shape of the array can be obtained with `shape`.

- Display the image using `imshow()`, but note that OpenCV uses BGR color ordering, unlike PIL's RGB.

- You can convert the color space using `cvtColor()`. For example, convert BGR to RGB.

- Converting an image to grayscale is also possible with `cvtColor()`.

- Save images using `imwrite()`.

- To work with color channels, you can use array slicing.

These are some basics of working with digital images in Python. In the next video, we'll explore more image processing techniques and tasks.

## Manipulating Images

**Title: Manipulating Images**

In this video, we will discuss manipulating digital images, including copying and flipping them.

**Copying Images**

Copying an image allows you to create a new image independent of the original. Consider the following image array. Using libraries like PIL or OpenCV, we can use the `id()` function to find the memory address of an object.

If we assign the "baboon" array to a variable "A," and then use the `id()` function to check its memory address, we see that it is the same as the original "baboon" array. This means "A" points to the same memory locations as the "baboon" array.

However, if we apply the `copy()` method to "baboon" and assign it to "B," we see that the memory address is different. "B" is now a copy of "baboon," and any changes made to "A" won't affect "B."

Here's a table displaying the image arrays and their corresponding memory addresses:

| Image Array | Memory Address |
|-------------|----------------|
| "Baboon"    | 0x123456       |
| "A"         | 0x123456       |
| "B"         | 0x789abc       |

If we set all elements in the "baboon" array to zero, using code like `baboon[:] = 0`, the "A" array will also change because it points to the same memory locations as "baboon." Both "baboon" and "A" will be zero.

However, because we used the `copy()` method, the "B" array remains unaffected.

You don't have to copy images all the time, but if you encounter this behavior in your code, it's essential to understand the difference.

**Flipping Images**

Flipping images changes their orientation. You can flip an image by changing the index values of pixels or intensities. Consider the following array:

```plaintext
1 2 3
4 5 6
7 8 9
```

If we convert the column indexes to row indexes, the image will have a different orientation:

```plaintext
1 4 7
2 5 8
3 6 9
```

For color images, we can flip all the color channels simultaneously.

**Using PIL for Flipping**

PIL (Pillow) provides several ways to flip an image:

1. You can use the `ImageOps` module and functions like `flip` or `mirror` to flip or mirror an image.
2. The `transpose()` method allows you to perform various flips, such as flipping vertically or horizontally.

**Using OpenCV for Flipping**

OpenCV offers several ways to flip an image:

1. The `flip()` function can flip an image. The `flipCode` parameter specifies the type of flip (0 for vertical flip, 1 for horizontal flip, and -1 for both).
2. The `rotate()` function can rotate the image. You can use predefined integers to specify the type of flip, like rotating 90 degrees clockwise.

In conclusion, copying and flipping images are essential operations in image processing. Both PIL and OpenCV provide various methods to perform these tasks, making it easier to manipulate and work with digital images.

## Manipulating Images One Pixel at a Time

**Image Indexing and Cropping:**
- Images can be represented as arrays with pixel intensities.
- Cropping involves selecting specific portions of an image using slicing on rows and columns.
- Cropping can be performed on multiple color channels.

**Cropping with PIL and OpenCV:**
- Vertical and horizontal cropping can be done using slicing.
- Demonstrations of cropping on NumPy arrays and PIL image objects.
- How to perform cropping operations using PIL and OpenCV.

**Changing Pixel Intensities:**
- Changing pixel intensities by setting specific array values.
- Drawing simple shapes like rectangles to modify images.

**Drawing Shapes and Text:**
- Using the PIL `ImageDraw` module to draw shapes and text on images.
- Parameters for drawing shapes, including coordinates, bounding boxes, and fill colors.
- Overlaying text on an image.

**Superimposing Images:**
- Superimposing one image onto another using coordinates specifying where to paste the image.
- Demonstrating the `paste()` method in PIL for superimposing images.

**OpenCV Pixel Manipulations:**
- Using OpenCV functions for pixel manipulations.
- Creating shapes and overlaying text using OpenCV's `rectangle` and `putText` functions.

## Pixel Transformation

In this video, the focus is on Pixel Transformations using OpenCV for grayscale images. The topics covered include Histograms, Intensity Transformations, Thresholding, and Simple Segmentation.

### Histograms
- A histogram counts the number of occurrences of pixel intensities in an image.
- Intensity values are represented as an array, and the histogram provides insights into the distribution of these values.
- Histograms are typically represented as bar graphs, with darker portions corresponding to lower intensities and brighter regions to higher intensities.

### Intensity Transformations
- Intensity transformations change an image pixel by pixel, mapping one intensity value to another.
- These transformations can shift and scale the histogram of the image.
- Image Negatives reverse the intensity levels, making image details more evident.

### Brightness and Contrast Adjustments
- Linear transforms can be used to adjust brightness and contrast.
- The linear model involves alpha (contrast control) and beta (brightness control).
- The `convertScaleAbs` function is used to apply these transformations, scaling and calculating absolute values to keep intensity values in the 0 to 255 range.
- Adjusting alpha and beta changes contrast and brightness, respectively.

### Histogram Equalization
- Histogram Equalization is an algorithm that improves contrast by flattening the image's histogram.
- It uses the image's histogram to determine a transform that enhances contrast.

### Thresholding
- Thresholding is used in segmentation to extract objects from an image.
- A threshold function applies a threshold to each pixel, categorizing them as foreground (usually 1) or background (usually 0 or 255).
- Thresholding is demonstrated with an example, where pixels exceeding a threshold are set to 255, and others to 0.

### Automatic Thresholding (OTSU)
- Sometimes, selecting a threshold manually can be challenging.
- The OTSU method automatically selects an optimal threshold value to segment an image.
- The image segmentation results using OTSU are compared to manual thresholding.

This video explores various techniques for transforming and enhancing grayscale images through histograms, intensity transformations, and segmentation using thresholding.

## Geometric Operations

In this video, Geometric Operations are explored, including scaling, translation, and rotation. These operations are applied to one-channel representations of images, but they can generally be applied to each channel simultaneously. The image is treated as a function of vertical (y) and horizontal (x) directions, with sampling occurring at integer points.

### Geometric Transformations
- Geometric transformations involve changing the coordinates (x and y) of the image.
- The resulting image, denoted as "g," is a function of new coordinates (x' and y').
- This video focuses on a subset of geometric transformations called Affine transformations.

### Scaling
- Scaling reshapes the image, either shrinking or expanding it along the horizontal and/or vertical directions.
- Scaling along the x-axis can be represented by a scaling factor "a."
- Scaling factor "a" of 2, for example, doubles the width of the image, stretching it.

### Interpolation
- When scaling, not all pixel values in the new image have corresponding values in the original image.
- Interpolation methods, such as nearest neighbor interpolation, are used to estimate pixel values based on neighboring pixels.

### Translation
- Translation involves shifting the image horizontally (tx) or vertically (ty).
- Pixels shifted out of the image bounds are typically replaced with zero values.
- To accommodate shifted pixels, the image size may need to be increased.

### Affine Transformation Matrix
- Geometric transformations can be represented as a matrix equation.
- The Affine Transformation matrix includes parameters for translation, scaling, and shearing (not covered in this video).
- OpenCV accepts this matrix as an input for transformations.

### Rotation
- Images can be rotated by an angle theta.
- A rotation matrix is used to perform counter-clockwise rotations.
- Simplifications can be made, assuming isotropic scaling and rotation from the image center.
- Libraries like PIL and OpenCV provide simple methods for image rotation.

### Practical Implementation
- In PIL (Python Imaging Library), you can resize an image by specifying the desired width or height.
- OpenCV provides functions like "resize" for scaling, "warpAffine" for translation, and "getRotationMatrix2D" for rotation.
- Translation and rotation matrices are applied to images using these functions.

## Spatial Operations in Image Processing

In this video, the topic of Spatial Operations in Image Processing is discussed. Spatial Operations involve various techniques such as Convolution (Linear Filtering), Edge Detection, and Median Filters. These operations are typically applied to each channel of an image independently.

### Convolution and Linear Filtering
- Convolution is a fundamental technique for filtering images, using a kernel or filter to apply a specific operation.
- Convolution involves overlaying a kernel on an image and performing element-wise multiplications followed by summation.
- The result of convolution is a new image, which can enhance certain characteristics of the original image.
- Convolution can be used to filter images and perform tasks like smoothing and sharpening.

### Padding
- To handle images of different sizes, padding techniques like zero-padding or value replication can be used to adjust image dimensions.

### Low Pass Filters
- Low Pass Filters, such as mean filters, are used to smooth images and reduce noise.
- These filters average pixel values within a neighborhood, resulting in a smoother image.
- There's a trade-off between sharpness and smoothness when using these filters.

### Edge Detection
- Edge Detection is crucial in computer vision and identifies areas where image brightness changes sharply.
- It approximates derivatives and gradients to find these edges.
- Sobel operators are used for horizontal and vertical derivative approximations.

### Median Filters
- Median Filters are effective at removing noise but may distort the image.
- They compute the median value within a neighborhood, replacing the central pixel value.

### Applying Spatial Operations in OpenCV
- In OpenCV, you can apply spatial operations to images.
- Techniques like mean filtering and image sharpening can be achieved using predefined functions.
- GaussianBlur is used to smooth images, and Sobel functions help compute derivatives.
- The magnitude of gradients can be calculated to represent edges in an image.

# Weel 3: Introduction to Image Classification

## Introduction to Image Classification

This video provides an overview of image classification, discussing its definition, applications, and challenges. Image classification is the process of automatically categorizing images into specific classes or labels, such as identifying objects like cats, cars, or buildings. It is widely used in various fields, from organizing smartphone photos to assisting medical professionals in radiology and aiding self-driving cars in navigating roads.

### Image Classification Basics

- Image classification starts with defining a set of categories or classes, like "cat" and "dog," represented as Y values (Y=0 for cat, Y=1 for dog).
- Computers interpret images through intensity values, typically in the form of digital images. In the case of RGB images, they are represented as three-dimensional arrays or tensors with consistent row and column dimensions.
- The dataset consists of images (X) and their corresponding labels (Y), where each image is associated with a class label. For instance, image X4 is labeled as Y4=1, indicating it's a dog.
- More complex datasets, like the MNIST database of handwritten digits, have multiple classes (0 to 9) and comprise small grayscale images of single digits (28x28 pixels).

### Challenges of Image Classification

Image classification poses various challenges, including:

1. **Change in Viewpoint:** Images may vary in perspective or orientation, making it challenging to recognize objects from different angles.

2. **Change of Illumination:** Differences in lighting conditions can impact an image's appearance and affect classification accuracy.

3. **Deformation:** Objects may undergo deformation, distortion, or changes in shape, making it harder to identify them.

4. **Occlusion:** Objects may be partially or fully obscured, hindering their recognition.

5. **Background Clutter:** Complex or cluttered backgrounds can interfere with accurate image classification.

The video suggests that, due to these challenges, the module will explore several supervised machine learning methods for image classification, such as K-Nearest Neighbors, feature extraction, and linear classifiers. While it introduces these concepts, the video does not delve into specific code implementations.

## Linear Classifiers

In the video titled "Linear Classifiers," the concept of linear classifiers is explained. Linear classifiers are fundamental in classification tasks and serve as the basis for more advanced classification methods. Here's a summary of the key points, including the mathematical equations:

### Two-Class Classification:
- The video focuses on the two-class classification problem, where images are labeled as either "cat" (y = 0) or "dog" (y = 1).
- Three-channel images can be concatenated into vectors for processing.

### Decision Plane and Decision Boundary:
- Linear classifiers use a decision plane, represented as an equation: 
  - Decision Plane: $$z = w^T * x + b$$
  - Decision Boundary: z = 0
- Here, "w" represents the weight vector, "x" represents the feature vector of the image, and "b" is the bias term. "w" and "b" represents learnable parameters.
- Anything on the left side of the decision boundary is classified as a dog, while anything on the right side is classified as a cat.

### Calculating Z and Threshold Function:
- The value of "z" is calculated for each sample, indicating its position relative to the decision boundary.
- A threshold function is used to convert "z" values into class labels:
  - Threshold Function: $$y hat = {1 if z > 0, 0 if z <= 0}$$

### Limitations of Linear Separability:
- Linear classifiers may not always effectively separate data, especially when data points are not linearly separable. Misclassifications can occur.

### Introduction to the Logistic Function:
- The logistic function, also known as the sigmoid function, is introduced as an alternative to the threshold function:
  - Logistic Function: $$σ(z) = 1 / (1 + e^(-z))$$
  - Where "e" is the base of the natural logarithm and "z" is the linear combination of weights and features.

### Determining Class Labels with Logistic Function:
- The output of the logistic function is used to determine class labels:
  - If the output > 0.5, y hat = 1 (dog).
  - If the output <= 0.5, y hat = 0 (cat).
- The sigmoid function's values range between 0 and 1, offering a probabilistic interpretation.

### Practical Application:
- Linear classifiers can be used in image classification applications. When given an image, the classifier calculates the probabilities and outputs the class.

### Conclusion:
- Linear classifiers are foundational in classification tasks, but they may have limitations in handling complex data.
- The logistic function provides a probabilistic approach to classification and can perform better than simple thresholding.

In summary, linear classifiers, including both threshold-based and logistic-based approaches, are explored in the context of two-class image classification. These methods provide a basis for understanding more advanced classification techniques, and the corresponding mathematical equations are provided to illustrate their implementation.

## Logistic Regression Training: Gradient Descent

In this video, titled "Logistic Regression Training: Gradient Descent," the process of training a logistic regression classifier is discussed, including the use of gradient descent for parameter optimization. Here are the key concepts with the corresponding mathematical equations:

### Training a Classifier:
- Training involves finding the best learnable parameters (weights "w" and bias "b") of the decision boundary.
- The decision boundary separates classes and is crucial for classification.

### Classification Loss and Cost:
- A loss function measures how good the prediction of the classifier is.
- Classification Loss (also known as 0-1 Loss):
  - $$Loss(y hat, y) = {0 if y hat = y, 1 if y hat ≠ y}$$
- The cost is the sum of the loss over all training samples:
  - Cost(w, b) = Σ Loss(y hat, y)

### Cross Entropy Loss:
- In practice, cross-entropy loss is used instead of classification loss.
- Cross Entropy Loss:
  - &&Loss(y hat, y) = - [y * log(y hat) + (1 - y) * log(1 - y hat)]$$
- The cost is still the sum of the loss over all training samples.

### Gradient Descent:
- Gradient descent is a method to find the minimum of the cost function.
- The gradient provides the slope of the cost function.
- The update equation for bias "b" using gradient descent is:
  - $$b_i+1 = b_i - η * ∂Cost(w, b) / ∂b$$
- η (eta) is the learning rate, a small positive number that controls the step size.

### Learning Rate Selection:
- Selecting the right learning rate is crucial for the convergence of gradient descent.
- A learning rate that is too small may result in slow convergence.
- A learning rate that is too large can lead to oscillations and failure to converge.
- The learning rate is a hyperparameter chosen based on validation data.

### Gradient Descent in Parameter Space:
- In higher dimensions with multiple parameters, gradient descent is performed on the entire parameter set.
- The gradient becomes a vector, and updates are made to both "w" and "b."

### Learning Curve:
- The learning curve is a plot of the cost function against the number of iterations.
- It shows how the cost decreases as the model iteratively updates parameters.
- Generally, more parameters require more data and iterations for the model to converge.

### Challenges of Gradient Descent:
- Gradient descent may face challenges with functions that have regions where the gradient is zero, leading to convergence issues.

In summary, this video explains the process of training a logistic regression classifier using gradient descent to find the optimal parameters that minimize the cost function. It also highlights the importance of selecting an appropriate learning rate for successful convergence.

## Mini-Batch Gradient Descent

In this video titled "Mini-Batch Gradient Descent," the concept of Mini-Batch Gradient Descent for training machine learning models is explained. This approach allows training with more data efficiently. The key points from the video are as follows:

### Introduction to Mini-Batch Gradient Descent:
- Mini-Batch Gradient Descent is a variation of gradient descent where only a subset of the training data is used in each iteration.
- Instead of using the entire dataset in one go, a few samples (mini-batch) are used for each iteration.
- It can be thought of as minimizing a mini cost function or total loss.

### Epochs and Batch Gradient Descent:
- When all samples in the dataset are used in one iteration, it's referred to as batch gradient descent, and one iteration equals one epoch.

### Mini-Batch Gradient Descent in Practice:
- Mini-Batch Gradient Descent uses a few samples (batch) to calculate the cost and update model parameters.
- The number of iterations required to complete one epoch depends on the batch size.
- For example, with a batch size of three, it takes three iterations to complete one epoch.

### Calculating the Number of Iterations:
- To determine the number of iterations for different batch sizes and epochs, you can divide the number of training examples by the batch size.
- For example, with a batch size of one, there are six iterations for a dataset of six samples.

### Monitoring and Overfitting:
- At the end of each epoch, the accuracy on validation data is calculated to monitor model performance.
- Overfitting occurs when the accuracy on the validation data starts to decrease, indicating that the model has been trained too much.

In summary, Mini-Batch Gradient Descent is a technique used for training machine learning models efficiently by using subsets of the training data in each iteration. The number of iterations depends on the batch size and is used to update model parameters. Monitoring accuracy on validation data helps prevent overfitting during training.

## SoftMax and Multi-Class Classification

**Transcript:** 

The argmax function is introduced, which returns the index of the largest value in a sequence of numbers. Practical examples illustrate its application.

### Handling Multi-Class Classification

Logistic regression, designed for two-class problems, is extended to address multi-class problems. Separate planes are used for each class, and equations represent these planes. For example:

- Class 0 (cat): $$\(Z_0 = w_0^TX + b_0\)$$
- Class 1 (dog): $$\(Z_1 = w_1^TX + b_1\)$$
- Class 2 (fish): $$\(Z_2 = w_2^TX + b_2\)$$

### SoftMax Function

The SoftMax function converts dot products into probabilities for multi-class classification. Probability of belonging to class \(i\) is calculated as:

$$\[P(Y=i|X) = \frac{e^{Z_i}}{\sum_{j=0}^{2} e^{Z_j}}\]$$

### Classification Process

Similar to logistic regression, SoftMax is employed for classification. Training SoftMax is almost identical to logistic regression.

### Alternative Methods

Alternative methods for creating multi-class classifiers are briefly mentioned, such as "one versus rest" and "one versus one," utilized in support vector machines.

## Support Vector Machines

### Introduction to SVM

Support Vector Machines (SVM) are discussed in this video, focusing on their application in classification tasks.

### Kernels and Transformations

- Kernels: SVMs are introduced as a tool for classification. Kernels play a crucial role in SVMs. A dataset is considered linearly separable if a plane can cleanly separate each class. However, not all datasets are linearly separable.
- Transforming Data: To deal with non-linearly separable data, data transformation is explained. Using a simple example with one feature, it's shown how data can be transformed into a higher-dimensional space to make it linearly separable.

### Kernel Functions

- Kernel Types: Different kernel functions are discussed, including:
  - Linear
  - Polynomial
  - Radial basis function (RBF)
- RBF Kernel: The RBF kernel is highlighted as the most widely used. It computes the difference between two inputs and involves a parameter called Gamma.

### Selecting Gamma

- Gamma Selection: The process of selecting the appropriate Gamma value is explained using a dataset of cats and dogs. Different Gamma values are tested, and the impact on classification accuracy is discussed. Overfitting is addressed when Gamma is too high.

### Validation Data

- Using Validation Data: To avoid overfitting, the importance of using validation data is emphasized. The dataset is split into training and validation sets to find the best value of Gamma.

### Maximum Margin

- Maximum Margin: SVMs aim to find a hyperplane that maximizes the margin between classes. The concept of support vectors is introduced, where only support vectors significantly affect the classification.
- Optimization: The optimization procedure for finding the optimized hyperplane is briefly mentioned, but complex mathematical details are skipped.

### Soft Margin SVM

- Soft Margin SVM: In cases where classes are not perfectly separable, the concept of the soft margin SVM is introduced. The regularization parameter (C) controls the allowance of misclassified samples.

### Hyperparameter Selection

- Hyperparameter Selection: The video concludes by highlighting the importance of selecting the best values for Gamma and the regularization parameter C based on their performance on the validation data.

##  Image Feature

### Introduction to Image Features

- Traditional Image Classification: Using image intensities for classification has limitations due to the sensitivity of pixel relationships to minor shifts in the image.
- Image-to-Vector Conversion: To address this, images are converted into vectors, often using large patches instead of individual pixel values.
- Pixel Relationships: Classification depends on pixel relationships, and even slight image shifts can alter the feature vector significantly.

### Feature Types

- Definition of Features: Features are measurements extracted from images to aid in classification.
- Color Histogram: An example of a feature is the color histogram, which counts intensity occurrences but doesn't consider pixel relationships.
- Sub-Images and Histograms: To address this limitation, images can be split into sub-images, and histograms can be calculated for each sub-image.
- Color Challenges: Color-based features may not always be suitable, as they might not capture the desired characteristics, such as for classifying shapes.

### Grayscale and Gradients

- Grayscale Conversion: Converting images to grayscale can reveal similarities in shapes.
- Surprising Gradients: Gradients in grayscale images can be identical, even for visually distinct shapes.

### Histogram of Oriented Gradients (HOG)

- HOG Overview: Histogram of Oriented Gradients (HOG) is introduced as an image feature.
- Calculation Process: HOG counts gradient orientation occurrences in localized image regions.
- Unit Circle Example: The concept of HOG is illustrated using a unit circle, where gradients are calculated.
- Gradients for Objects: Gradients are used to generate histograms for objects, capturing their unique characteristics.

### HOG Feature Extraction

- Image Processing: A practical example involves converting an image to grayscale, calculating gradient magnitudes and angles using Sobel operators.
- Grid Division: Images are divided into a grid of cells.
- Histograms in Cells: For each cell, a histogram of gradient directions is created.
- Block Normalization: To handle imbalances in lighting, cells are block normalized.
- Final HOG Vector: The HOG feature vector combines pixel-level histograms and is used with SVM for image classification.
- Parameter Considerations: The example simplifies the process, but parameters like the number of cells and angle bins must be considered.

### Other Image Features

- SURF and SIFT: Mention of other image features like SURF and SIFT is made, with reference to the OpenCV documentation for more information.

### Machine Learning Process Summary

- Overview: The machine learning process is summarized, including feature extraction, non-linear mapping (Kernel), and linear classification.

# Week 4: Neural Networks and Deep Learning for Image Classification

##  Neural Networks

### Introduction to Neural Networks

- Dataset: A non-linearly separable dataset in one dimension is presented.
- Decision Function Analogy: Classification in neural networks is likened to a decision function, where values are mapped to one or zero on the vertical axis.
- Box Function: An example of a decision function resembling a box function is shown.
- Neural Network Approximation: Neural networks aim to approximate such functions using learnable parameters.

### Neural Network Representation

- Logistic Regression Comparison: Neural networks can be seen as approximating box functions using logistic regression.
- Cat-Dog Dataset: A cat-dog dataset is used as an example where a straight line cannot separate the data.
- Nodes and Edges: Nodes represent the line, while edges represent inputs and outputs.
- Activation Function: The logistic function applied in neural networks is referred to as the activation function.
- Sigmoid Function: The sigmoid function is illustrated, and its output is called the activation.
- Incorrect Results: Applying the sigmoid function to some data points results in incorrect classifications.

### Function Subtraction

- Subscripted Sigmoid Functions: Two sigmoid functions, "A sub script one" and "A sub script two," are introduced.
- Function Subtraction: Subtracting the second sigmoid function from the first approximates the desired decision function.
- Linear Function: The linear function is applied, subtracting the second activations from the first activation.
- Thresholding: A threshold is applied, mapping values less than 0.5 to zero and greater than 0.5 to one, yielding the desired function.

### Neural Network Architecture

- Neural Network Structure: The process is depicted graphically, showing two linear functions and sigmoid activations.
- Hidden Layer: The hidden layer contains two artificial neurons, while the output layer has one neuron.
- Artificial Neurons: Each linear function and activation is called an artificial neuron.

### Multi-Dimensional Input

- Input Dimension: Additional dimensions can be added to the input, resulting in more weights between layers.
- Learnable Parameters: Neural networks have a substantial number of learnable parameters, with modern networks having millions.
- Fully Connected Networks: Such networks are often called Feedforward Neural Networks or fully connected networks.

### Multi-Dimensional Classification

- Multi-Dimensional Dataset: A non-linearly separable dataset in two dimensions is presented.
- Dimension-Dependent Neurons: The number of neurons depends on the input dimensions.
- Decision Function Plot: The decision function is visualized in two dimensions, mapping cats to zero and dogs to one.

## Convolutional Networks

### Introduction to CNNs

- **CNN Overview:** Convolutional Neural Networks (CNNs) are a type of neural network designed for image classification. They consist of various layers, including convolution, pooling, and fully connected layers, which extract features and classify objects within images.

### Feature Extraction with Convolution

- **Feature Learning:** CNNs extract features from input images. These feature extraction layers are analogous to the feature learning layers in CNNs.
  
- **Convolution Operation:** Convolution layers use learnable parameters called kernels (filters). The convolution operation can be expressed as follows for a single channel (grayscale) input:

  - \( \text{Output} = \sigma(W * \text{Input} + b) \)

    Where:
    - \( \sigma \) represents an activation function (e.g., ReLU).
    - \( W \) is the kernel.
    - \( * \) denotes the convolution operation.
    - \( b \) is a bias term.

- **Activation Maps:** The output of a convolution layer is an activation map or feature map, similar to a one-channel image. Each kernel detects different properties or features of the input.

### Stacking Convolution Layers

- **Multiple Kernels:** In CNNs, multiple kernels (neurons) are applied to the input, resulting in multiple feature maps. For example, if there are \( M \) kernels, there will be \( M \) feature maps.

- **Stacking Layers:** CNNs can stack convolution layers. The output of one layer becomes the input to the next. Neurons are replaced with kernels, and the process is repeated.

### Receptive Field

- **Receptive Field:** The receptive field is the region in the input that influences a single pixel in the activation map. A larger receptive field captures more information about the entire image.

- **Increasing Receptive Field:** To increase the receptive field, CNNs can add more layers, which requires fewer parameters than increasing kernel size.

### Pooling

- **Pooling Operation:** Pooling layers reduce the number of parameters, increase the receptive field, and preserve essential features. Max pooling, a popular pooling method, involves taking the maximum value from a specified region.

### Flattening and Fully Connected Layers

- **Flattening:** After feature extraction, CNNs flatten or reshape the output to create a feature vector. If the output of a max pooling layer is \( 7 \) units wide and \( 7 \) units high, it's flattened into a vector.

- **Fully Connected Layer:** The flattened output serves as the input to fully connected layers. Each neuron in a fully connected layer has input dimensions equal to the flattened output's size.

- **Example:** If there are \( 32 \) output channels, each \( 4 \times 4 \) in size, the flattened output will have \( 512 \) elements ( \( 32 \times 16 \) ). Each neuron in the fully connected layer will have \( 512 \) input dimensions.

The equations and explanations provided here illustrate how CNNs build features, stack layers, handle receptive fields, use pooling, and process flattened data for fully connected layers. These mathematical insights help us understand the inner workings of CNNs and their feature extraction capabilities. 

##  CNN Architecture

### Introduction to CNN Architectures

- **Popular CNN Architectures:** There are several popular CNN architectures, including LeNet-5, AlexNet, VGGNet, and ResNet, each designed for various image classification tasks.

### LeNet-5

- **LeNet-5 Overview:** LeNet-5, proposed by Yann LeCun in 1989, was one of the earliest CNNs. It was particularly successful in recognizing handwritten digits in the MNIST dataset.

- **Input and Convolution:** LeNet-5 receives a grayscale image as input and applies a 5x5 filter with a stride of 1, resulting in a 28x28 output volume.

- **Pooling Layer:** A pooling layer follows, producing 14x14 outputs. The network repeats this pattern of convolution and pooling layers.

- **Fully Connected Layers:** It eventually reaches fully connected layers, where the output is flattened into 120 neurons, followed by another layer with 84 neurons. The sigmoid activation function is applied to produce the final output.

### Rise of CNNs

- **Comparison and ImageNet:** CNNs gained prominence in image classification when they outperformed other methods on benchmark datasets like ImageNet. For example, in 2012, AlexNet achieved a record-breaking accuracy of 63.3%.

### AlexNet

- **AlexNet Architecture:** AlexNet consists of multiple layers with various parameters. It has convolutional layers with different kernel sizes.

- **Large Number of Parameters:** The first convolutional layers have 11x11 kernels and 25 channels, resulting in a substantial number of parameters.

### VGGNet

- **VGGNet Introduction:** VGGNet, a Very Deep Convolutional Network, was designed to reduce the number of parameters and improve training time.

- **Variants:** VGGNet comes in multiple variants, such as VGG-19 and VGG-16, indicating the number of layers.

- **Key Insight:** VGGNet introduced the idea of replacing larger kernels with stacked 3x3 convolution layers to maintain the receptive field while reducing parameters and computations.

### ResNet

- **ResNet for Deeper Networks:** As CNNs grew deeper, the vanishing gradient problem emerged. ResNet introduced residual learning, where skip connections allow gradients to bypass different layers, enabling the construction of much deeper networks.

- **ResNet Example:** A 32-layer network from the Deep Residual Learning for Image Recognition paper exemplifies the concept.

### Transfer Learning

- **Transfer Learning Concept:** Transfer learning involves using pre-trained CNNs to classify images instead of building a network from scratch.

- **Replacing SoftMax Layer:** In the simplest form, you replace the SoftMax layer of the pre-trained model with your SoftMax layer, adjusting the number of neurons for your specific classification task.

- **SVM Usage:** Depending on your dataset's size, you can opt to use Support Vector Machines (SVM) instead of the SoftMax layer. Pretrained models serve as feature generators in this context.

Understanding these mathematical and architectural aspects of CNNs and transfer learning can be invaluable for leveraging CNNs in various image classification tasks.

# Week 5

## Object Detection

### Sliding Windows for Object Detection

- **Sliding Windows Algorithm:** Object detection involves locating multiple objects within an image, often using a systematic approach like the sliding windows algorithm.

- **Window Classification:** Consider a fixed window size for detection. If the chosen window size can contain the object of interest, it's classified as the object (e.g., a dog), while other windows are classified as background.

- **Systematic Approach:** The sliding windows algorithm systematically moves across the image, classifying sub-images within each window.

- **Shift and Repeat:** After classifying one region, the window shifts, and the process repeats. Horizontal border traversal may lead to a vertical shift.

### Challenges in Object Detection

- **Overlapping Detections:** Object detectors often produce multiple overlapping detections, leading to redundancy.

- **Object Size Variability:** Objects may come in various sizes within the same image, requiring strategies like resizing or reshaping.

- **Shape Variability:** The same object can have different shapes in images, necessitating similar adjustments.

- **Overlapping Objects:** Overlapping objects in pictures can pose challenges for sliding windows-based methods.

### Bounding Boxes for Object Detection

- **Bounding Box Concept:** Bounding boxes are rectangular boxes used for object detection. They are defined by coordinates such as `(y_min, x_min)` for the upper-left corner and `(y_max, x_max)` for the lower-right corner.

- **Predicted Bounding Boxes:** In object detection, the goal is to predict these bounding box coordinates, typically represented with a "hat" symbol to indicate they are predictions.

### Bounding Box Pipeline

- **Components:** Similar to image classification, object detection involves classes (`y`) and bounding boxes (`x`).

- **Training:** Models for object detection are trained on datasets containing class labels and corresponding bounding box coordinates.

- **Model Prediction:** When an image with objects to detect is input into the trained model, it predicts both the class and bounding box coordinates.

### Score-Based Image Classification

- **Confidence Score:** Many object detection algorithms provide a score to indicate the model's confidence in its prediction.

- **Score Range:** Scores typically range from 0 to 1, with higher values indicating greater confidence.

- **Thresholding:** Detection results can be filtered based on score thresholds. Only detections with scores above a certain threshold are accepted.

- **Example:** For example, if a dog is predicted with a high score (e.g., 0.99), the model is highly confident. If the score is low (e.g., 0.5), the model lacks confidence in the prediction.

- **Threshold Adjustment:** By adjusting the score threshold, you can control the number of accepted detections.

- **Redundancy Mitigation:** Thresholding helps mitigate redundancy in overlapping detections.

In this video, we're going to explore the Haar feature-based Cascade Classifiers method for object detection, including the mathematical concepts behind it.

## Object Detection with Haar Cascade Classifier

### Introduction to Haar Cascade Classifier

- **Objective:** The Haar feature-based Cascade Classifier is used for detecting objects like cars, traffic lights, pedestrian stop signs, and more in images.

- **Method Overview:** The approach was proposed by P. Viola and M. Jones in 2001. It's a machine learning method that involves training a cascade function on a large dataset of both positive images (containing the object of interest) and negative images (representing the background).

### Haar Wavelets and Feature Extraction

- **Haar Wavelets:** The Haar feature classifier relies on Haar wavelets, which are convolution kernels used to extract various types of features from images. These features include edges, lines, and diagonal edges.

- **Feature Extraction:** During training, millions of images are processed by the classifier to extract relevant features using Haar wavelets. These features play a crucial role in object detection.

- **Feature Overlay:** An example is shown where Haar wavelets are overlaid on an image of a car, highlighting the edges and features extracted by these wavelets.

### Integral Image Concept

- **Integral Image:** The integral image is a fundamental concept used in the Haar Cascade Classifier. It's a way to represent an image where each pixel contains the cumulative sum of the corresponding input pixels above and to the left of that pixel.

- **Calculating Integral Sum:** To calculate the integral sum for a pixel, you add the values to its left and above it. If there's nothing to the left or above, the sum is the pixel's value itself.

- **Example:** For a highlighted pixel, the integral sum is obtained by adding values from multiple directions, ultimately resulting in a cumulative sum.

### AdaBoost for Feature Selection

- **Feature Selection:** The algorithm selects important features from a large set to create highly efficient classifiers. This is achieved through the use of AdaBoost (Adaptive Boosting), a machine learning algorithm.

- **AdaBoost Concept:** AdaBoost assigns weights to both classifiers and samples. It focuses on observations that are difficult to classify correctly, selecting only features that improve classifier accuracy.

- **Reduction in Features:** In the example of a 24x24 window, initially, over 180,000 features are generated. However, AdaBoost significantly reduces this number to approximately 6,000 relevant features.

### Cascade of Classifiers

- **Cascades:** The Haar Cascade Classifier employs cascades of classifiers, which group sub-images from the input image in stages and disregard regions that don't match the object being detected.

- **Sub-Image Classification:** At each stage, the classifier determines whether the sub-image corresponds to the object of interest. If not, the sub-window is discarded along with its associated features.

- **Progressive Stages:** The process continues through multiple stages, with each stage refining the classification. Only sub-images that pass all stages are considered as detections.

Understanding these mathematical and algorithmic components is crucial for comprehending how Haar Cascade Classifiers work for object detection.
