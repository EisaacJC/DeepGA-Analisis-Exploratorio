# Índice de contenidos
  
- 1. [DeepGA Análisis Exploratorio](#id1)
   - a. [Resumen](#id1-a)
- 2. [Ejecución](#ejec)
- 3. [Evaluación de DeepGA](#id2)
  - a.[MNIST](#id2-a)
  - b. [Fashion-MNIST](#id2-b)
- 4.[Comparativa del estado del arte](#id3)
  - a. [MNIST](#id3-a)
  - b. [Fashion-MNIST](#id3-b)
- 5. [Referencias] (#ref) 


# DeepGA-Analisis-Exploratorio<a name="id1"></a>
## Resumen <a name="id1-a"></a>
Se presenta una comparativa entre el algoritmo de neuro evolución DeepGA y redes neuronales convolucionales del estado del arte obteniendo para los dos conjuntos de imágenes MNIST y Fashion-MNIST mejores valores de precisión y una menor cantidad del número de parámetros con el uso de DeepGA obteniendo una precisión promedio 98.63±0.20% para el conjunto MNIST y de 90.80±0.53% para el conjunto Fashion-MNIST. Se observa una ventaja en el porcentaje de clasificación en ambos conjuntos mediante el uso de DeepGA correspondientes a valores que oscilan entre (0.29%-53.41%) para MNIST y de (5.76%-42.11%) para Fashion-MNIST. El número de parámetros obtenido para el conjunto MNIST fue de 24104±5188.54 y para FashionMNIST 56687.6±11929.7.
# Ejecución en Google Colaboratory<a name="ejec"></a>
Para llevar a cabo la evaluación del algoritmo DeepGA en los conjuntos de datos mencionados previamente, se llevaron a cabo 5 ejecuciones de 20 generaciones con los parámetros mostrados en la Tabla 1.  La ejecución se llevó a cabo en el entorno GPU de Google Colaboratory en su versión paga, la cual provee de una GPU Nvidia-P100 de 16 GB.  
Para la evaluación de DeepGA se elige un porcentaje de 30% del conjunto de datos para hacer la validación y un 70% del mismo para llevar a cabo el entrenamiento. 

Para poder llevar a cabo la ejecución se recomienda subir las carpetas DeepGA_Digits y DeepGA_Fashion a google Drive al igual que los archivos fashion.zip y  digits.zip, posteriormente subir las libretas de Jupyter y finalmente ejecutar las mismas. 

- Un aspecto importante a considerar es el tiempo de ejecución por conjunto de datos, cada ejecución demora alrededor de 8 horas usando Google Colaboratory Pro con la tarjeta gráfica Nvidia-P600 de 16 GB de capacidad. 

# Evaluación de DeepGA en MNIST y Fashion-MNIST<a name="id2"></a>
## MNIST<a name="id2-a"></a>
### Gráfica de función de aptitud, precisión y número de parámetros<a name="id2-aa"></a>
![image](https://user-images.githubusercontent.com/10681481/176562187-60a2c7a5-7f7f-43c6-a00f-86b54c88f9d0.png)
### Evaluación de DeepGA<a name="id2-ab"></a>
![image](https://user-images.githubusercontent.com/10681481/176562296-be560383-b52c-449a-98eb-bb82a18d9935.png)

## Fashion-MNIST<a name="id2-b"></a>
### Gráfica de función de aptitud, precisión y número de parámetros<a name="id2-ba"></a>
![image](https://user-images.githubusercontent.com/10681481/176562264-20cd3883-f863-4176-8cc7-a93d020ddfb0.png)
### Evaluación de DeepGA<a name="id2-bb"></a>
![image](https://user-images.githubusercontent.com/10681481/176562372-2fdd2b74-8742-4346-b8c9-35b6bfd0e741.png)

#Comparativa contra modelos del estado del arte<a name="id3"></a>
## MNIST<a name="id3-a"></a>
![image](https://user-images.githubusercontent.com/10681481/176562564-531dc8b8-cc86-4cc3-b551-1e671c7ee28f.png)
## Fashion-MNIST<a name="id3-b"></a>
![image](https://user-images.githubusercontent.com/10681481/176562596-baf410f8-72e7-47b8-9e7b-23affcd81539.png)

# Referencias principales:<a name="ref"></a>

- 1.	He, K., Zhang, X., Ren, S., & Sun, J. (2012). Deep Residual Learning for Image Recognition. Retrieved June 27, 2022, from https://image-net.org/challenges/LSVRC/2015/
- 2.	Krizhevsky, A., & Inc, G. (2014). One weird trick for parallelizing convolutional neural networks.
- 3.	Simonyan, K., & Zisserman, A. (2015). VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION. http://www.robots.ox.ac.uk/
- 4. Verbancsics, P., & Harguess, J. (2015). Image classification using generative neuro evolution for deep learning. Proceedings - 2015 IEEE Winter Conference on Applications of Computer Vision, WACV 2015, 488–493. https://doi.org/10.1109/WACV.2015.71
- 5. Wang, B., Sun, Y., Xue, B., & Zhang, M. (2019). A Hybrid GA-PSO Method for Evolving Architecture and Short Connections of Deep Convolutional Neural Networks. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 11672 LNAI, 650–663. https://doi.org/10.1007/978-3-030-29894-4_52/COVER/
- 6.	Vargas, G. A., Eng, H. B., Gabriel, H., & Mesa, A. (2021). Neuroevolution of Convolutional Neural Networks for COVID-19 Classification in X-ray Images.
- 7. Yann LeCun, Corinna Cortes, & Chris Burges. (n.d.). MNIST handwritten digit database. Retrieved June 28, 2022, from http://yann.lecun.com/exdb/mnist/
- 8. Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. https://trends.google.com/trends/explore?date=all&q=mnist,CIFAR,ImageNet
- 9. Vargas Hákim Gustavo Adolfo. (2021). DeepGA-GitHub. Retrieved June 28, 2022, from https://github.com/GustavoVargasHakim/DeepGA
- 10.	Sun, Y., Xue, B., Zhang, M., Yen, G. G., & Lv, J. (2020). Automatically Designing CNN Architectures Using the Genetic Algorithm for Image Classification. IEEE Transactions on Cybernetics, 50(9), 3840–3854. https://doi.org/10.1109/TCYB.2020.2983860
- 11.	Gustavo Adolfo Vargas Hákim. (2021). Neuroevolution of Convolutional Neural Networks for COVID-19 Classification in X-ray Images
