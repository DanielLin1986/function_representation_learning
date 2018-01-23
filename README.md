## Function Representation Learning for Vulnerability Discovery

Hi there, welcome to this page!

The page contains the code and data used in the paper [Vulnerability Discovery with Function Representation Learning from Unlabeled Projects](https://dl.acm.org/citation.cfm?id=3138840) by Guanjun Lin, Jun Zhang, Wei Luo, Lei Pan and Yang Xiang.

### Requirements:

 * [Tensorflow](https://www.tensorflow.org/)
 * [Keras](https://github.com/fchollet/keras/tree/master/keras)
 * Python >= 2.7
 * [CodeSensor](https://github.com/fabsx00/codesensor)

The dependencies can be installed using [Anaconda](https://www.anaconda.com/download/). For example:

```bash
$ bash Anaconda3-5.0.1-Linux-x86_64.sh
```

### Instructions:

The Vulnerabilities_info.xlsx file contains information of the collected function-level vulnerabilities. These vulnerabilities are from 3 open source projects: [FFmpeg](https://github.com/FFmpeg/FFmpeg), [LibTIFF](https://github.com/vadz/libtiff) and [LibPNG](https://github.com/glennrp/libpng). And vulnerability information was collected from [National Vulnerability Database(NVD)](https://nvd.nist.gov/) until the mid of July 2017.

The "Data" folder contains the source code of vulnerable functions and non-vulnerable functions within the Zip file of the 3 projects. After unzipping the files, one will find that the source code of each vulnerable function was named with its CVE ID. For the non-vulnerable functions, they were named with "{filename}_{functionname}.c" format. 

The "Code" folder contains the Python code samples. 
1) ProcessCFilesWithCodeSensor.py file is for invoking the CodeSensor to parse functions to ASTs in serialized format (for detail information and usage of CodeSensor, please visit the author's blog: http://codeexploration.blogspot.com.au/ for more details). 
2) ProcessRawASTs_DFT.py file is to process the output of ProcessCFilesWithCodeSensor.py and convert the serialized ASTs to textual vectors.
3) BlurProjectSpecific.py file is to blur the project specific content and convert the textual vectors (the output of ProcessRawASTs_DFT.py) to numeric vectors which can be used as the input of ML algorithms. 
4) LSTM.py file contains the Python code sample for implementing LSTM network based on Keras with Tensorflow backend.

We used [Understand](https://scitools.com/) which is a commercial code enhancement tool for extracting function-level code metrics. In CodeMetrics.xlsx file, we include 23 code metrics extracted from the vulnerable functions of 3 projects. 

### Possible Future Work

1) In our paper, we randomly selected one code metric which was the "essential complexity" as the proxy (used as the substitute of the actual label). It will be interesting to examine whether the performance can be further improved when combining multiple code metrics, since multiple code metrics can provide more information and are more indicative of potential vulnerability (i.e. overly complex code are difficult to understand, therefore harder to debug and maintain).

2) The proposed LSTM network structure is fairly simple. We believe that the performance can be improved by implementing more complex network structure. For instance, adding pooling layers and/or dropout. One can even try the attention mechanism with LSTM. 

### Contact:

You are welcomed to improve our code as well as our method. Please kindly cite our paper if you use the code/data in your work. For acquiring more data or inquiries, please contact: junzhang@swin.edu.au.

Thanks!
