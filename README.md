# heygem-face2face-source
heygem 合成视频逆向源码

# 模型下载
`bash download.sh`

安装 **Python 3.8**。然后，使用 pip 安装项目依赖项  
```bash
conda create -n heygem python=3.8 -y # 创建环境
conda activate heygem # 进入环境
python -V #查看是否为3.8版本
```
安装cuda相关配置
```bash
conda install cudatoolkit=11.8 cudnn=8.9.2 #安装cuda 11.8版本
pip install onnxruntime-gpu==1.16.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
然后输入以下命令安装
```bash

pip install -r requirements.txt 
# 指定阿里云镜像
# pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

```
安装ffmpeg等软件依赖，输入以下命令安装 ffmpeg版本一定要高于4.0
```
sudo apt update
sudo apt install ffmpeg -y
```
# 启动接口版本
`python app.py`

# 启动web版本
`python gradio_web.py`
---

![image](https://github.com/user-attachments/assets/0964d61b-fc3a-4922-8481-1cba270602e8)


硅基不让我卖源码了
如果需要源码请自行破解，相关逆向技术可以沟通交流



