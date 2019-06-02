import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = [3,3]
FLATTEN_SIZE = [3,3,3] #[IMAGE_SIZE, Channels=3]
MAX_RATING = 5;
EPOCHS = 5;

def initTensorFlow():
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  tf.enable_eager_execution()
  return initModel()
  

def initModel():
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=FLATTEN_SIZE),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(MAX_RATING, activation=tf.nn.softmax)
  ])

  model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  return model

def trainModel(model, data, labels):
  # train_data & train_labels = type numpy.ndarray
  four_fifths =int(len(data) * 0.8);
  train_data = data[0: four_fifths]
  train_labels = labels[0: four_fifths]
  test_data = data[four_fifths: len(data)]
  test_labels = labels[four_fifths: len(data)]

  model.fit(train_data, train_labels, epochs = EPOCHS)
  test_loss, test_acc = model.evaluate(test_data, test_labels)
  print('Test accuracy:', test_acc)

def modelPredict(model, imagePath):
  model.predict(preprocess_image(path))


def preprocess_image(path):
  image = tf.read_file(path)
  image = tf.image.decode_jpeg(image, channels=0)
  image = tf.image.resize(image, IMAGE_SIZE)
  image /= 255.0  # normalize to [0,1] range
  return image.numpy()

#def resizeImages(all_images):
  

