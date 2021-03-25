# Introduction

CNN is currently one of the most well-known techniques in inverse image reconstruction problems. It has been proved to be successful in a large number of tasks, including image denoising, single-image super-resolution and reconstructing lossy images.

<!-- Réseau neuronal convolutif (en anglais ***CNN***) est actuellement l'une des techniques les plus connues dans les problèmes de reconstruction d'image inverse. Il s'est avéré efficace dans un grand nombre de tâches, y compris le débruitage d'image, la super-résolution d'image et la reconstruction d'images avec perte. -->

This architecture's power is attributed to their ability to learn from many large sets of images. However, [-@2007.02471; -@1711.10925] and many other works have demonstrated that the architecture of a CNN can act as a sufficiently strong prior to enable image reconstruction, even without any training data. Specifically, un-trained networks perform well for denoising image [-@2007.02471], compressive sensing [-@1806.06438], and even for reconstructing videos [-@1910.01684].

<!-- La puissance de cette architecture est attribuée à sa capacité à apprendre à partir de nombreux grands ensembles d'images. Cependant, [-@2007.02471; -@1711.10925] et de nombreux autres travaux ont démontré que l'architecture d'un CNN peut agir comme un préalable suffisamment fort pour permettre la reconstruction d'image, même sans aucune donnée d'apprentissage. Plus précisément, les réseaux non formés fonctionnent bien pour le débruitage de l'image [-@2007.02471], l'acquisition comprimée [-@1806.06438] et même pour la reconstruction de vidéos [-@1910.01684].-->

In this work, we will focus on this technique (***Deep Image Prior***). First, we will take a look at some examples. Seconds, we go deeper into the underlying principle of it through the previous example. Lastly, we will experiment what we have just presented above and do some further studies.

<!-- Dans ce travail, nous nous concentrerons sur cette technique (***Deep Image Prior***). Tout d'abord, nous allons examiner quelques exemples. En second lieu, nous approfondissons le principe sous-jacent de celui-ci à travers l'exemple précédent. Enfin, nous expérimenterons ce que nous venons de présenter ci-dessus et ferons quelques études complémentaires.-->
