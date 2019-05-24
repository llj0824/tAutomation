import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = [3, 3]
MAX_RATING = 5;
EPOCHS = 5;

def initTensorFlow():
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  tf.enable_eager_execution()
  return initModel()
  

def initModel():
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=IMAGE_SIZE),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(MAX_RATING, activation=tf.nn.softmax)
  ])

  model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  return model

def trainModel(model, train_data, train_labels):
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
  return image



