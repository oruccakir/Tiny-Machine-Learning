import tensorflow as tf

# Load in fashion MNIST
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Define the base model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)), 
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
"""
Neural Networks learn the best when the data is scaled / normalized to fall in a constant range.
One practitioners often use is the range [0,1]. How might you do this to the training and test images used here?

A hint: these images are saved in the standard RGB format
"""

training_images  = training_images / 255 #YOUR CODE HERE#
test_images = test_images / 255#YOUR CODE HERE#

"""
Using these improved images lets compile our model using an adaptive optimizer to learn faster and a categorical loss function to differentiate between the the various classes 
we are trying to classify. Since this is a very simple dataset we will only train for 5 epochs.
"""

# compile the model
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fit the model to the training data
model.fit(training_images, training_labels, epochs=5)

# test the model on the test data
model.evaluate(test_images, test_labels)

"""
Once it's done training -- you should see an accuracy value at the end of the final epoch. It might look something like 0.8648. This tells you that your neural network is about 86% accurate in classifying the training data. I.E., it figured out a pattern match between the image and the labels that worked 86% of the time. But how would it work with unseen data? That's why we have the test images. We can call model.evaluate, and pass in the two sets, and it will report back the loss for each. This should reach about .8747 or thereabouts, showing about 87% accuracy. Not Bad!

But what did it actually learn? If we inference on the model using model.predict we get out the following list of values. What does it represent?

A hint: trying running print(test_labels[0])
"""

classifications = model.predict(test_images)
print(classifications[0])


"""
Let's now look at the layers in your model. What happens if you double the number of neurons in the dense layer. 
What different results do you get for loss, training time etc? Why do you think that's the case?
"""

NUMBER_OF_NEURONS = 1024

# define the new model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(NUMBER_OF_NEURONS, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# compile fit and evaluate the model again
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)

"""
Consider the effects of additional layers in the network instead of simply more neurons to the same layer. 
First update the model to add an additional dense layer into the model between the two existing Dense layers.
"""

YOUR_NEW_LAYER = tf.keras.layers.Dense(512,activation=tf.nn.relu)

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    YOUR_NEW_LAYER,
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

"""
Lets then compile, fit, and evaluate our model. What happens to the error? How does this compare to the original model and the model with double the number of neurons?
"""

# compile fit and evaluate the model again
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)


"""
Before you trained, you normalized the data. What would be the impact of removing that? 
To see it for yourself fill in the following lines of code to get a non-normalized set of data and then re-fit and evaluate the model using this data.
"""

# get new non-normalized mnist data
training_images_non = training_images * 255#YOUR_CODE_HERE#
test_images_non = test_images * 255 #YOUR_CODE_HERE#

# re-compile, re-fit and re-evaluate
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    YOUR_NEW_LAYER,
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images_non, training_labels, epochs=5)
model.evaluate(test_images_non, test_labels)
classifications = model.predict(test_images_non)


"""
Sometimes if you set the training for too many epochs you may find that training stops improving and you wish you could quit early. Good news, you can! TensorFlow has a function called Callbacks which can check the results from each epoch. Modify this callback function to make sure it exits training early but not before reaching at least the second epoch!

A hint: logs.get(METRIC_NAME) will return the value of METRIC_NAME at the current step
"""

# define and instantiate your custom Callback
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if( logs.get("accuracy") > 0.86):
      self.model.stop_training = True
callbacks = myCallback()

# re-compile, re-fit and re-evaluate
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                            tf.keras.layers.Dense(512, activation=tf.nn.relu),
                            YOUR_NEW_LAYER,
                            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer = tf.keras.optimizers.Adam(),
      loss = 'sparse_categorical_crossentropy',
      metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])