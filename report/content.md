# Introduction

CNN is currently one of the most well-known techniques in inverse image reconstruction problems. It has been proved to be successful in a large number of tasks, including image denoising, single-image super-resolution and reconstructing lossy images.

<!-- Réseau neuronal convolutif (en anglais ***CNN***) est actuellement l'une des techniques les plus connues dans les problèmes de reconstruction d'image inverse. Il s'est avéré efficace dans un grand nombre de tâches, y compris le débruitage d'image, la super-résolution d'image et la reconstruction d'images avec perte. -->

This architecture's power is attributed to their ability to learn from many large sets of images. However, [@2007.02471; @1711.10925] and many other works have demonstrated that the architecture of a CNN can act as a sufficiently strong prior to enable image reconstruction, even without any training data. Specifically, un-trained networks perform well for denoising image [@2007.02471], compressive sensing [@1806.06438], and even for reconstructing videos [@1910.01684].

<!-- La puissance de cette architecture est attribuée à sa capacité à apprendre à partir de nombreux grands ensembles d'images. Cependant, [@2007.02471; @1711.10925] et de nombreux autres travaux ont démontré que l'architecture d'un CNN peut agir comme un préalable suffisamment fort pour permettre la reconstruction d'image, même sans aucune donnée d'apprentissage. Plus précisément, les réseaux non formés fonctionnent bien pour le débruitage de l'image [@2007.02471], l'acquisition comprimée [@1806.06438] et même pour la reconstruction de vidéos [@1910.01684].-->

In this work, we will focus on this technique (***Deep Image Prior***). First, we will take a look at some examples. Seconds, we go deeper into the underlying principle of it through the previous example. Lastly, we will experiment what we have just presented above and do some further studies.

<!-- Dans ce travail, nous nous concentrerons sur cette technique (***Deep Image Prior***). Tout d'abord, nous allons examiner quelques exemples. En second lieu, nous approfondissons le principe sous-jacent de celui-ci à travers l'exemple précédent. Enfin, nous expérimenterons ce que nous venons de présenter ci-dessus et ferons quelques études complémentaires.-->

# Les aspects techniques

## Le code

First, we go into the code. As said above, we tried to rewrite the code of U. Since each problem requires a different model, we only focus on the Denoising Model ( although our code should work with most models in that paper ). Beside the reasons mentioned above, there are some technical reasons for our motivations of refactoring U’s code:

- The code is outdated ( it were first written in 2017 ).
- Mixing between Pytorch Sequential and Module API, which makes the code hard to follow (especially the model-building function ).
- Irritating model by raw loop, which makes integrating with modern API and tools ( e.g Keras Callback, TensorBoard ) more difficult.

We rewrite it using TensorFlow, and we also add additional metrics for measuring the PSNR between the denoised image and both noisy and original image.

<!--
Tout d'abord, nous entrons dans le code. Comme indiqué ci-dessus, nous avons essayé de réécrire le code dans [@1711.10925]. Étant donné que chaque problème nécessite un modèle différent, nous nous concentrons uniquement sur le modèle de débruitage (bien que notre code devrait fonctionner avec la plupart des modèles de cet article).

Outre la motivation de mieux comprendre le code, il y a quelques raisons techniques à nos motivations pour un réusinage:

- Le code est obsolète (il a été écrit pour la première fois en 2017).
- Mélange entre Pytorch Sequential et Module API, ce qui rend le code difficile à suivre (en particulier la fonction de création de modèle).
- Modèle irritant par boucle brute, ce qui rend plus difficile l'intégration avec des API et des outils modernes (par exemple TensorBoard).

Nous le réécrivons à l'aide de TensorFlow, et nous ajoutons également des métriques supplémentaires pour mesurer le PSNR entre l'image débruitée et l'image bruyante et le PSNR entre l'image débruitée et l'image originale.
-->

## Le modèle

Here, we see a very typical CNN architecture. Although U model is deeper and has a lot more layer, it still has the same group of layers  as an autoencoder ( decoding and encoding ) with skip connections.

The model's input is a tensor with a shape of [batch_size, image_height, image_width, 32] where batch_size is 1 in our case. 32 is the number of channels ( features ) which will be explained later. Its output is a tensor with a shape of [batch_size, image_height, image_width, 3] which could be displayed as a image.

<!-- Ici, nous voyons une architecture CNN très typique. Bien que le modèle U soit plus profond et ait beaucoup plus de couches, il a toujours le même groupe de couches qu'un auto-encodeur (décodage et encodage) avec sauts de connexions.

L'entrée du modèle est un tenseur de forme [batch_size, image_height, image_width, 32] où batch_size vaut 1 dans notre cas. 32 est le nombre de canaux (fonctionnalités) qui seront expliqués plus loin. Sa sortie est un tenseur avec une forme de [batch_size, image_height, image_width, 3] qui pourrait être affiché comme une image.-->

## Décodage et encodage

Decoding is a process of extracting features from an image. On the opposite side, encoder will collect the features from the decoders and rebuild the image. Features could be anything, vary from edges, color, to resolution.

In our case, the number of 32 from the previous section means that we capture 32 differents features from noisy image to reconstruct the original image. That explains why if we iterrate the process too many time, the output image will tend to the noisy image. Because at a certain step,  the decoders will consider the noise as a feature and capture it.

Under the hood, decoder is built from convolutional operations. By traversing each pixel and applying the convolutional kernel, we could extract desired features. The illustration of a convolutional operations in image could be seen as follow:

<!-- Le décodage est un processus d'extraction de caractéristiques d'une image. Sur le côté opposé, l'encodeur collectera les caractéristiques des décodeurs et reconstruira l'image. Les caractéristiques peuvent être n'importe quoi, varient des bords, de la couleur à la résolution.

Dans notre cas, le nombre de 32 de la section précédente signifie que nous capturons 32 caractéristiques différentes à partir d'une image bruyante pour reconstruire l'image d'origine. Cela explique pourquoi si nous répétons le processus trop souvent, l'image de sortie tendra vers l'image bruyante. Parce qu'à une certaine étape, les décodeurs considéreront le bruit comme une caractéristique et le captureront.

Sous le capot, le décodeur est construit à partir d'opérations convolutives. En parcourant chaque pixel et en appliquant le noyau convolutif, nous pourrions extraire les fonctionnalités souhaitées. L'illustration d'une opération convolutive en image peut être vue comme suit:-->

## La sortie

We could see clearly the differences between each epoch. This could be explained as follow:

- In the first epoch, the input is a random noise and the model is initialized with random weights. So we see the a random image, which isn't related to our desired output.
- Around the 1000th epoch, the decoders inside model start to adjust itself to extract the features from the noisy image. First, they extract the most important ones like edge, color, resolution, etc.
- At the 3000th epoch, the decoders and encoders are now able to extract all the desired characteristics and reconstructing an output image as seen in
- The more we loop though the process, the more parts are extracted and undesired characteristics like noise are also included. Therefore, we see an image that is quite close to the noisy image.

<!--Nous pouvions voir clairement les différences entre chaque époque. Cela pourrait s'expliquer comme suit:

- A la première époque, l'entrée est un bruit aléatoire et le modèle est initialisé avec des poids aléatoires. Nous voyons donc une image aléatoire, qui n'est pas liée à la sortie souhaitée.
- Vers la 1000e époque, les décodeurs à l'intérieur du modèle commencent à s'ajuster pour extraire les caractéristiques de l'image bruyante. Premièrement, ils extraient les plus importants comme le bord, la couleur, la résolution, etc.
- A la 3000ème époque, les décodeurs et encodeurs sont désormais capables d'extraire toutes les caractéristiques souhaitées et de reconstruire une image de sortie comme on le voit dans
- Plus nous bouclons le processus, plus il y a de parties extraites et des caractéristiques indésirables comme le bruit sont également incluses. Par conséquent, nous voyons une image assez proche de l'image bruyante.
-->
