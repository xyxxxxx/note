[playground](https://katacoda.com/courses/kubernetes/playground)提供了包含了两个节点（Master 和一个 Node）的k8s集群，可用于测试。

```shell
# 集群信息
$ kubectl cluster-info

# get 获取有关资源的信息
$ kubectl get pod

$ kubectl get pod -o wide

$ kubectl get deployments
# NAME                  READY   UP-TO-DATE   AVAILABLE   AGE
# kubernetes-bootcamp   1/1     1            1           32s
# READY 表示 现存的/需求的 实例数
# UP-TO-DATE 表示处于就绪状态的实例数
# AVAILABLE 表示现有多少实例数可用

$ kubectl get rs
# NAME                             DESIRED   CURRENT   READY   AGE
# kubernetes-bootcamp-765bf4c7b4   1         1         1       5m18s
# DESIRED 表示应用需求的实例数
# CURRENT 表示正在运行的实例数
# READY 表示准备就绪的实例数

# describe 获取有关资源的详细信息
$ kubectl describe pod

# logs 打印 pod 和其中容器的日志
$ kubectl logs $POD_NAME

# exec 在 pod 中的容器上执行命令
$ kubectl exec $POD_NAME env

# 使 apiserver 监听本地的 8001 端口
$ kubectl proxy --port=8001
# Starting to serve on 127.0.0.1:8001
```



## Minikube 演示

```shell
# 1. (再)启动 Minikube 并创建一个集群
$ minikube start
# 😄  minikube v1.13.0 on Ubuntu 18.04
# ✨  Using the virtualbox driver based on existing profile
# 👍  Starting control plane node minikube in cluster minikube
# 🔄  Restarting existing virtualbox VM for "minikube" ...
# ...
# 🐳  Preparing Kubernetes v1.19.0 on Docker 19.03.12 ...
# ...
# 🔎  Verifying Kubernetes components...
# 🌟  Enabled addons: default-storageclass, storage-provisioner
# 🏄  Done! kubectl is now configured to use "minikube" by default


# 2. 创建一个 deployment
$ kubectl create deployment hello-minikube --image=k8s.gcr.io/echoserver:1.10
# 使用 Nodejs 镜像
# deployment.apps/hello-minikube created

# 3. 查看 deployment 状态
$ kubectl get deployment

# 4. 将 deployment 作为 service 公开
$ kubectl expose deployment hello-minikube --type=NodePort --port=8080
# service/hello-minikube exposed

# 5. 查看 pod 状态
$ kubectl get pod
# NAME                              READY   STATUS    RESTARTS   AGE
# hello-minikube-5d9b964bfb-x7f5j   1/1     Running   0          93s

# 6. 查看 service 状态
$ kubectl get services

# 7. 获取 service 的 url
$ minikube service hello-minikube --url
# http://192.168.99.100:32481
```

```
Hostname: hello-minikube-5d9b964bfb-x7f5j
xinxi
Pod Information:
	-no pod information available-

Server values:
	server_version=nginx: 1.13.3 - lua: 10008

Request Information:
	client_address=172.17.0.1
	method=GET
	real path=/
	query=
	request_version=1.1
	request_scheme=http
	request_uri=http://192.168.99.100:8080/

Request Headers:
	accept=text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
	accept-encoding=gzip, deflate
	accept-language=en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,ja-JP;q=0.6,ja;q=0.5,zh-TW;q=0.4
	connection=keep-alive
	host=192.168.99.100:32481
	upgrade-insecure-requests=1
	user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36

Request Body:
	-no body in request-
```

```shell
# 查看当前支持的插件
$ minikube addons list

# 启动插件
$ minikube addons enable metrics-server

# 禁用插件
$ minikube addons disable metrics-server
```

```shell
# 8. 删除 service
$ kubectl delete services hello-minikube
# service "hello-minikube" deleted

# 9. 删除 deployment
$ kubectl delete deployment hello-minikube
# deployment.extensions "hello-minikube" deleted

# 10. 停止本地 Minikube 集群
$ minikube stop
# ✋  Stopping node "minikube"  ...
# 🛑  1 nodes stopped.

```



## 在容器上执行命令

```shell
# 在指定 POD 的容器（唯一容器？）上启动 console
$ kubectl exec -it $POD_NAME bash

# 退出 console
$ exit
```



## 使用 label

```shell
# 将 deployment 作为 service 公开
$ kubectl expose deployment/kubernetes-bootcamp --type="NodePort" --port 8080
# service/kubernetes-bootcamp exposed

$ kubectl get services
# NAME                  TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
# kubernetes            ClusterIP   10.96.0.1       <none>        443/TCP          35s
# kubernetes-bootcamp   NodePort    10.111.29.188   <none>        8080:30985/TCP   1s

# 查看 label
$ kubectl describe services/kubernetes-bootcamp
# ...
# Labels:                   run=kubernetes-bootcamp
# ...

$ kubectl describe deployment
# ...
# Labels:                 run=kubernetes-bootcamp
# ...

# 为 pod 添加新 label
$ kubectl label pod $POD_NAME app=v1
# pod/kubernetes-bootcamp-765bf4c7b4-m4qb7 labeled

# 查看 pod 的 label
$ kubectl describe pods $POD_NAME
# ...
# app=v1
# pod-template-hash=765bf4c7b4
# run=kubernetes-bootcamp
# ...

# 通过 label 参数获取 pod 信息
$ kubectl get pods -l app=v1
# NAME                                   READY   STATUS    RESTARTS   AGE
# kubernetes-bootcamp-765bf4c7b4-m4qb7   1/1     Running   0          15m
```



## 扩容应用

```shell
$ kubectl get deployment
# NAME                  READY   UP-TO-DATE   AVAILABLE   AGE
# kubernetes-bootcamp   1/1     1            1           32s

# 查看 deployment 创建的实例集
$ kubectl get rs
# NAME                             DESIRED   CURRENT   READY   AGE
# kubernetes-bootcamp-765bf4c7b4   1         1         1       5m18s

# 为 deployment 设定实例数
$ kubectl scale deployments/kubernetes-bootcamp --replicas=4
# deployment.apps/kubernetes-bootcamp scaled

$ kubectl get deployments
# NAME                  READY   UP-TO-DATE   AVAILABLE   AGE
# kubernetes-bootcamp   4/4     4            4           10m

$ kubectl get pod -o wide
# NAME                                   READY   STATUS    RESTARTS   AGE   IP NODE       NOMINATED NODE   READINESS GATES
# kubernetes-bootcamp-765bf4c7b4-fgs8h   1/1     Running   0          78s   172.18.0.9 minikube   <none>           <none>
# kubernetes-bootcamp-765bf4c7b4-jw5n8   1/1     Running   0          78s   172.18.0.8 minikube   <none>           <none>
# kubernetes-bootcamp-765bf4c7b4-ljtmp   1/1     Running   0          10m   172.18.0.2 minikube   <none>           <none>
# kubernetes-bootcamp-765bf4c7b4-rkg2t   1/1     Running   0          78s   172.18.0.7 minikube   <none>           <none>

# 每次请求调度的 pod 不同，表示负载均衡正在工作
$ curl $(minikube ip):$NODE_PORT
# Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-765bf4c7b4-fgs8h | v=1
$ curl $(minikube ip):$NODE_PORT
# Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-765bf4c7b4-jw5n8 | v=1

```



## 更新应用

```shell
$ kubectl get pods
# NAME                                   READY   STATUS    RESTARTS   AGE
# kubernetes-bootcamp-765bf4c7b4-2mv5p   1/1     Running   0          30s
# kubernetes-bootcamp-765bf4c7b4-5bdb2   1/1     Running   0          30s
# kubernetes-bootcamp-765bf4c7b4-pn8gz   1/1     Running   0          30s
# kubernetes-bootcamp-765bf4c7b4-q9vqz   1/1     Running   0          30s

$ kubectl describe pods
# Name:         kubernetes-bootcamp-765bf4c7b4-2mv5p
# Namespace:    default
# Priority:     0
# Node:         minikube/172.17.0.31
# ...
# Labels:       pod-template-hash=765bf4c7b4
#               run=kubernetes-bootcamp
# Annotations:  <none>
# Status:       Running
# IP:           172.18.0.8
# IPs: ...
# Containers:
#   kubernetes-bootcamp:
#     Container ID:    docker://4855e2940e9526068749e48fd07798ca355cc57baf70d35c120ee52909aa22c9
#     Image:          gcr.io/google-samples/kubernetes-bootcamp:v1
#     Image ID:       docker-pullable://jocatalin/kubernetes-bootcamp@sha256:0d6b8ee63bb57c5f5b6156f446b3bc3b3c143d233037f3a2f00e279c8fcc64af
# ...

# 更新 deployment 的应用镜像
$ kubectl set image deployment/kubernetes-bootcamp kubernetes-bootcamp=jocatalin/kubernetes-bootcamp:v2
# deployment.apps/kubernetes-bootcamp image updated

$ kubectl get pods
# NAME                                   READY   STATUS        RESTARTS   AGE
# kubernetes-bootcamp-765bf4c7b4-2mv5p   1/1     Terminating   0          50s
# kubernetes-bootcamp-765bf4c7b4-5bdb2   1/1     Terminating   0          50s
# kubernetes-bootcamp-765bf4c7b4-pn8gz   1/1     Terminating   0          50s
# kubernetes-bootcamp-765bf4c7b4-q9vqz   1/1     Terminating   0          50s
# kubernetes-bootcamp-7d6f8694b6-h4bqg   1/1     Running       0          6s
# kubernetes-bootcamp-7d6f8694b6-lb5ss   1/1     Running       0          3s
# kubernetes-bootcamp-7d6f8694b6-q7c67   1/1     Running       0          6s
# kubernetes-bootcamp-7d6f8694b6-tnmzg   1/1     Running       0          3s

$ kubectl describe pod
# Name:         kubernetes-bootcamp-7d6f8694b6-h4bqg
# Namespace:    default
# Priority:     0
# Node:         minikube/172.17.0.31
# ...
# Labels:       pod-template-hash=7d6f8694b6
#               run=kubernetes-bootcamp
# Annotations:  <none>
# Status:       Running
# IP:           172.18.0.11
# IPs: ...
# Containers:
#   kubernetes-bootcamp:
#     Container ID:   docker://ba4095a12d1371681e6a2df406893390529c34558b92992c3f4382b3cd40339d
#     Image:          jocatalin/kubernetes-bootcamp:v2
#     Image ID:       docker-pullable://jocatalin/kubernetes-bootcamp@sha256:fb1a3ced00cecfc1f83f18ab5cd14199e30adc1b49aa4244f5d65ad3f5feb2a5
# ...

# 更新成功
$ curl $(minikube ip):$NODE_PORT
# Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-7d6f8694b6-q7c67 | v=2
$ curl $(minikube ip):$NODE_PORT
# Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-7d6f8694b6-h4bqg | v=2
```

```shell
# 再次更新 deployment 的应用镜像
$ kubectl set image deployment/kubernetes-bootcamp kubernetes-bootcamp=gcr.io/google-samples/kubernetes-bootcamp:v10
# deployment.apps/kubernetes-bootcamp image updated

$ kubectl get deployment
# NAME                  READY   UP-TO-DATE   AVAILABLE   AGE
# kubernetes-bootcamp   3/4     2            3           18m
# 2个 pod 更新，3个 pod 可用

$ kubectl get pod
# NAME                                   READY   STATUS             RESTARTS   AGE
# kubernetes-bootcamp-7d6f8694b6-h4bqg   1/1     Running            0          18m
# kubernetes-bootcamp-7d6f8694b6-q7c67   1/1     Running            0          18m
# kubernetes-bootcamp-7d6f8694b6-tnmzg   1/1     Running            0          18m
# kubernetes-bootcamp-886577c5d-hxmk9    0/1     ImagePullBackOff   0          52s
# kubernetes-bootcamp-886577c5d-lgjs6    0/1     ImagePullBackOff   0          53s

# 不存在 v10 镜像，拉取失败
$ kubectl describe pod
# ...
# Events:
#  Type     Reason     Age                    From               Message
#  ----     ------     ----                   ----               -------
#  Normal   Pulling    2m55s (x4 over 4m17s)  kubelet, minikube  Pulling image "gcr.io/google-samples/kubernetes-bootcamp:v10"
#  Warning  Failed     2m54s (x4 over 4m16s)  kubelet, minikube  Failed to pull image "gcr.io/google-samples/kubernetes-bootcamp:v10": rpc error: code = Unknown desc = Error response from daemon: manifest for gcr.io/google-samples/kubernetes-bootcamp:v10 notfound: manifest unknown: Failed to fetch "v10" from request "/v2/google-samples/kubernetes-bootcamp/manifests/v10".
#  Warning  Failed     2m54s (x4 over 4m16s)  kubelet, minikube  Error: ErrImagePull
#  Warning  Failed     2m29s (x6 over 4m15s)  kubelet, minikube  Error: ImagePullBackOff
#  Normal   BackOff    2m15s (x7 over 4m15s)  kubelet, minikube  Back-off pulling image "gcr.io/google-samples/kubernetes-bootcamp:v10"

# 回滚
$ kubectl rollout undo deployments/kubernetes-bootcamp
# deployment.apps/kubernetes-bootcamp rolled back

$ kubectl get pods
# NAME                                   READY   STATUS    RESTARTS   AGE
# kubernetes-bootcamp-7d6f8694b6-c8s6l   1/1     Running   0          31s
# kubernetes-bootcamp-7d6f8694b6-h4bqg   1/1     Running   0          25m
# kubernetes-bootcamp-7d6f8694b6-q7c67   1/1     Running   0          25m
# kubernetes-bootcamp-7d6f8694b6-tnmzg   1/1     Running   0          25m
```

