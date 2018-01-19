import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
#Comprobamos la version de tf en la maquina
if tf.__version__ < '1.4.0':
  raise ImportError('Version de tensorflow demasiado antigua, necesitas v1.4.* o posterior')

# Necesitamos esta libreria para que muestre las imagenes.
get_ipython().run_line_magic('matplotlib', 'inline')

# Esta linea es para que el prg llegue a la carpeta donde esta guardado todo.
sys.path.append("..")

#Importamos los dos archivos para la deteccion de objetos
from utils import label_map_util
from utils import visualization_utils as vis_util

#Modelo utilizado
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# Aqui se guardan los strings de las palabras para mostrar en los cuadros
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
	
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#Las imagenes deberan llamarse image1.jpg, image2.jpg... y estar en la carpeta test_images 
PATH_TO_TEST_IMAGES_DIR = 'test_images'
#Para meter más imagenes hay que cambiar el range. Ahora coge 3.
#Con 3 fotos tarda 15 min mas menos. Mejor no meter más porque aumenta demasiado el tiempo
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4) ]
#Aqui podemos cambiar el tamaño de las imagenes que muetsra el prg
IMAGE_SIZE = (11, 9)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definimos los tensores de entrada y salida para detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Esto es para las cajas que recuadran los objetos encontrados en cada foto.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Aqui guardamos el porcentaje de "acierto" que tiene el prg al detectar un objeto y lo mostramos al lado del nombre del objeto
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    #Abrimos las imagenes y las analizamos
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # Usaremos este array para los resultados
      image_np = load_image_into_numpy_array(image)
      # Ajustamos las dimensiones de las fotos
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Aqui analiza los objetos de la foto.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualizamos los resultados.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)

