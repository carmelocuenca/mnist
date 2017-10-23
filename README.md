# Un ejemplo de CNN para MNIST con *tensorflow*

Ver el libro "Learning TensorFlow" para todos los detalles

![CNN](getfile.png "A first example of CNN")


La capa de entrada corresponde a una imagen de 28x28 píxeles en b/w.

La operación realizada a esta capa de entrada son 32 convoluciones con una
máscara de dimensiones 5x5, que origina una capa  de 28x28x32; a este resultado
intermedio es aplicado un filtro no lineal *relu* con 32 *bias* diferentes dando
lugar a la capa de salida.

La siguiente capa intermedia es de menor dimensiones 14x14x32, y es generada
mediante un *max pool* que elige el máximo de una vecindad de 2x2.

 Añadimos otra capa más a partir de 64 filtros de convolución con una máscara
 3D de dimensiones 5x5x32. La capa intermedia resultante tiene dimensiones
 14x14x64.

 Otra capa de *max pool* de 2x2 reduce la dimensión de la capa intermedia a 7x7x64.

 A partir de esta capa intermedia obtenemos otra en forma de vector lineal con
 7x7x64 componentes.

 La última capa oculta es una capa *fully connected*

 La capa de salida es 1x10.

 ## Capa de entrada 28x28
 ```python
 # placeholder, a tensor that may be used as a handle for feeding a value,
 # but not evaluated directly
 x = tf.placeholder(tf.float32, shape=[None, 784])
 y_ = tf.placeholder(tf.float32, shape=[None, 10])

 # reshape returns a tensor that has the same values as tensor with shape shape
 # If one component of shape is the special value -1,
 # the size of that dimension is computed so that the total size remains constant
 x_image = tf.reshape(x, [-1, 28, 28, 1])
 ```

 Las imágenes de entrada vienen como vector *raw* en este ejemplo de 784 píxeles
 (28x28), así `x` contendrá el dato de entrada y `x_image` es realmente la capa
 de entrada. El vector `y_` representa la capa de salida.

 ## Capa interna 28x28x5 (convolución + relu)
La capa interna de nombre `conv1` la crea la rutina `conv_layer`:

```python
conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
```

la cual a su vez, crea dos nodos *tensorflow* de *variables* `W` y `b`.
`W` inicializado con una distribución normal, y `b` a un valor constante.
Construye el grafo para realizar las convoluciones espaciales y aplica una función
no lineal de salida *relu* con 32 *bias* diferentes. [Ver código](mnist.py) para más detalles.

## Capa intermedia 14x14x2 (*max_pool*)
Una simple reducción de detalles para reducir el volumen de neuronas y también
disminuir la sensibilidad de la red a cambios de localización de los detalles.

 ```python
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
# other things
conv1_pool = max_pool_2x2(conv1)
```

De acuerdo con la información de la función [max_pool](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool) en la web de *tensorflow*. `ksize` determina el tamaño de la ventana para calcular el máximo y `strides` el avance de la ventana.

## Capa intermedia de 14x14x64 (*convolución + relu*)
```python
conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
```


## Capa intermedia 7x7x64 (*max_pool*)
```python
conv2_pool = max_pool_2x2(conv2)
```

## Capa intermedia de 1D con 7x7x64 píxeles (*reshape*)
Simplemente un *reshape* del tensor

```python
conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
```

## Capa intermedia 1D de 1024 + (*full connected + relu*)
```python
def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b
# other things
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
```

## Capa intermedia 1D (*dropout*)
Evita el problema del *overfitting*

```python
keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
```

## Capa de salida (*dropout*)
y_conv = full_layer(full1_drop, 10)
