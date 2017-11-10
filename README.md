## Function Representation Learning

Hi there, welcome to this pape!

The page contains the code and data used in the paper [Vulnerability Discovery with Function Representation Learning from Unlabeled Projects](https://) by Guanjun Lin, Jun Zhang, Wei Luo, Lei Pan and Yang Xiang.

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

The "Data" folder contains the source code of vulnerable functions within the Zip file of the 3 projects. After unzipping the files, one will find that the source code of each function was named with its CVE ID. 

The "Code" folder contains the Python code sample for invoking the CodeSensor to parse functions to ASTs (for detail information and usage of CodeSensor, please visiter the author's blog: http://codeexploration.blogspot.com.au/) It also contains the Python code sample for implementing LSTM based on Keras with Tensorflow backend.

We used [Understand](https://scitools.com/) which is a commercial code enhancement tool for extracting function-level code metrics. In CodeMetrics.xlsx file, we include 23 code metrics extracted from the vulnerable functions of 3 projects. 

### Contact:

You are welcomed to improve our code as well as our method. Please cite our paper if you use the code/data in your work. For acquiring more data or enquiries, please contact: junzhang@swin.edu.au.

Thanks and enjoy coding!

