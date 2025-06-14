FROM nvidia/cuda:11.6.1-base-ubuntu20.04

# 还需要一个chk.tar.gz
WORKDIR /code
COPY . /code


RUN pip install --upgrade -r /code/requirements.txt  -i  https://pypi.tuna.tsinghua.edu.cn/simple/ --timeout 600

# 6、运行服务
#CMD ["uvicorn", "http_api2:app", "--host", "0.0.0.0", "--port", "80"]
EXPOSE  22
EXPOSE  80
CMD /usr/sbin/sshd -D
