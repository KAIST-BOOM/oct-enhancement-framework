# Deep learning-based image enhancement in optical coherence tomography by exploiting interference fringe
Optical coherence tomography (OCT), an interferometric imaging technique, provides non-invasive, high-speed, high-sensitive volumetric biological imaging in vivo. However, systemic features inherent in the basic operating principle of OCT limit its imaging performance such as spatial resolution and signal-to-noise ratio. Here, we propose a deep learning-based OCT image enhancement framework that exploits raw interference fringes to achieve further enhancement from currently obtainable optimized images. The proposed framework for enhancing spatial resolution and reducing speckle noise in OCT images consists of two separate models: an A-scan-based network (NetA) and a B-scan-based network (NetB). NetA utilizes spectrograms obtained via short-time Fourier transform of raw interference fringes to enhance axial resolution of A-scans. NetB was introduced to enhance lateral resolution and reduce speckle noise in B-scan images. The individually trained networks were applied sequentially. We demonstrate the versatility and capability of the proposed framework by visually and quantitatively validating its robust performance. Comparative studies suggest that deep learning utilizing interference fringes can outperform the existing methods. Furthermore, we demonstrate the advantages of the proposed method by comparing our outcomes with multi-B-scan averaged images and contrast-adjusted images. We expect that the proposed framework will be a versatile technology that can improve performance and functionality of OCT.

![image](https://user-images.githubusercontent.com/125458136/220089979-e9286413-823b-4260-9e0b-2dd0ddee3751.png)
Blind testing performance of deep learning-based OCT image enhancement framework. Left and right columns represent currently optimized input and enhanced final output, respectively. Note that, in these results, the currently optimized OCT data were the input. Final output is the result of sequentially applying both NetA and NetB to the input. Each figure represents a cucumber, b grape, c,d thyroid tissue specimen, e finger nail, f Scotch tape, g pork meat, h droplet with TiO2 microspheres, and i,j arterial cross-section. The ROIs (red and yellow boxes) on the right side of the image show magnified views (3X for a-g and i-j; 5X for h). Scale bars, 1 mm.
