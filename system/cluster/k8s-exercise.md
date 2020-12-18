[toc]

# 查看集群信息

[playground](https://katacoda.com/courses/kubernetes/playground)提供了包含了两个节点（Master 和一个 Node）的k8s集群，可用于测试。

```shell
# 集群信息
$ kubectl cluster-info

# get 获取有关资源的信息
$ kubectl get pod

$ kubectl get pod -o wide

$ kubectl get deployments
NAME                  READY   UP-TO-DATE   AVAILABLE   AGE
kubernetes-bootcamp   1/1     1            1           32s
# READY 表示 现存的/需求的 实例数
# UP-TO-DATE 表示处于就绪状态的实例数
# AVAILABLE 表示现有多少实例数可用

$ kubectl get rs
NAME                             DESIRED   CURRENT   READY   AGE
kubernetes-bootcamp-765bf4c7b4   1         1         1       5m18s
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
Starting to serve on 127.0.0.1:8001
```





# Minikube 演示

```shell
# 1. (再)启动 Minikube 并创建一个集群
$ minikube start
😄  minikube v1.13.0 on Ubuntu 18.04
✨  Using the virtualbox driver based on existing profile
👍  Starting control plane node minikube in cluster minikube
🔄  Restarting existing virtualbox VM for "minikube" ...
# ...
🐳  Preparing Kubernetes v1.19.0 on Docker 19.03.12 ...
# ...
🔎  Verifying Kubernetes components...
🌟  Enabled addons: default-storageclass, storage-provisioner
🏄  Done! kubectl is now configured to use "minikube" by default

# 2. 创建一个 deployment
$ kubectl create deployment hello-node --image=k8s.gcr.io/echoserver:1.4
# 创建的Pod根据指定Docker镜像运行container
deployment.apps/hello-node created

# 3. 查看 deployment 状态
$ kubectl get deployment
NAME         READY   UP-TO-DATE   AVAILABLE   AGE
hello-node   1/1     1            1           1m

# 4. 查看 pod 状态
$ kubectl get pod
NAME                          READY   STATUS    RESTARTS   AGE
hello-node-7567d9fdc9-6lkbf   1/1     Running   0          2m

# 5. 查看集群事件
$ kubectl get event
LAST SEEN   TYPE     REASON              OBJECT                             MESSAGE
3m56s       Normal   Scheduled           pod/hello-node-7567d9fdc9-6lkbf    Successfully assigned default/hello-node-7567d9fdc9-6lkbf to minikube
3m56s       Normal   Pulled              pod/hello-node-7567d9fdc9-6lkbf    Container image "k8s.gcr.io/echoserver:1.4" already present on machine
3m56s       Normal   Created             pod/hello-node-7567d9fdc9-6lkbf    Created container echoserver
3m56s       Normal   Started             pod/hello-node-7567d9fdc9-6lkbf    Started container echoserver
# 创建一个Pod
3m57s       Normal   SuccessfulCreate    replicaset/hello-node-7567d9fdc9   Created pod: hello-node-7567d9fdc9-6lkbf
# 将Deployment(replica set)的replica设为1
3m57s       Normal   ScalingReplicaSet   deployment/hello-node              Scaled up replica set hello-node-7567d9fdc9 to 1

# 默认情况下，Pod 只能通过 Kubernetes 集群中的内部 IP 地址访问
# 要使得 hello-node 容器可以从外部访问，必须将 Pod 暴露为 Service
# 6. 将 Deployment 作为 Service 公开
$ kubectl expose deployment hello-node --type=LoadBalancer --port=8080
service/hello-node exposed

# 7. 查看 service 状态
$ kubectl get service
NAME         TYPE           CLUSTER-IP     EXTERNAL-IP   PORT(S)          AGE
hello-node   LoadBalancer   10.97.131.53   <pending>     8080:31948/TCP   7s
kubernetes   ClusterIP      10.96.0.1      <none>        443/TCP          92d

# 8. 打开访问 service 的浏览器窗口
$ minikube service hello-node
```

```
CLIENT VALUES:
client_address=172.17.0.1
command=GET
real path=/
query=nil
request_version=1.1
request_uri=http://192.168.99.100:8080/

SERVER VALUES:
server_version=nginx: 1.10.0 - lua: 10001

HEADERS RECEIVED:
accept=text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
accept-encoding=gzip, deflate
accept-language=zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,ja-JP;q=0.6,ja;q=0.5,zh-TW;q=0.4
connection=keep-alive
host=192.168.99.100:31948
upgrade-insecure-requests=1
user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36
BODY:
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
# 9. 删除 service
$ kubectl delete service hello-node
service "hello-node" deleted

# 10. 删除 deployment
$ kubectl delete deployment hello-node
deployment.apps "hello-node" deleted

# 11. 停止本地 Minikube 集群
$ minikube stop
✋  Stopping node "minikube"  ...
🛑  1 nodes stopped.

```





# [学习 Kubernetes 基础知识](https://kubernetes.io/zh/docs/tutorials/kubernetes-basics/)——交互式教程

```shell
# 1. (再)启动 Minikube 并创建一个集群
$ minikube start
😄  minikube v1.15.1 on Ubuntu 18.04
# ...
🏄  Done! kubectl is now configured to use "minikube" by default

# 查看集群信息
$ kubectl cluster-info
# Master的位置
Kubernetes master is running at https://172.17.0.115:8443
# 集群各资源的url
KubeDNS is running at https://172.17.0.115:8443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.

# 查看Node, Pod信息
$ kubectl get node
NAME       STATUS   ROLES    AGE     VERSION
minikube   Ready    master   1m      v1.19.4  # 1个Master节点

$ kubectl get pod
No resources found in default namespace.      # 无Pod

# 2. 创建deployment,指定测试镜像
$ kubectl create deployment kubernetes-bootcamp --image=gcr.io/google-samples/kubernetes-bootcamp:v1
deployment.apps/kubernetes-bootcamp created

# 查看Deployment, Pod信息
$ kubectl get deployments
NAME                  READY   UP-TO-DATE   AVAILABLE   AGE
kubernetes-bootcamp   1/1     1            1           1m

$ kubectl describe deployments
Name:                   kubernetes-bootcamp
Namespace:              default
CreationTimestamp:      Tue, 15 Dec 2020 08:43:32 +0000
Labels:                 run=kubernetes-bootcamp
Annotations:            deployment.kubernetes.io/revision: 1
Selector:               run=kubernetes-bootcamp
Replicas:               1 desired | 1 updated | 1 total | 1 available | 0 unavailable
StrategyType:           RollingUpdate
MinReadySeconds:        0
RollingUpdateStrategy:  25% max unavailable, 25% max surge
Pod Template:                        # Pod的创建模板
  Labels:  run=kubernetes-bootcamp
  Containers:                        # 容器列表
   kubernetes-bootcamp:
    Image:        gcr.io/google-samples/kubernetes-bootcamp:v1
    Port:         8080/TCP
    Host Port:    0/TCP
    Environment:  <none>
    Mounts:       <none>
  Volumes:        <none>
Conditions:
  Type           Status  Reason
  ----           ------  ------
  Available      True    MinimumReplicasAvailable
  Progressing    True    NewReplicaSetAvailable
OldReplicaSets:  <none>
NewReplicaSet:   kubernetes-bootcamp-765bf4c7b4 (1/1 replicas created)
Events:
  Type    Reason             Age   From                   Message
  ----    ------             ----  ----                   -------
  Normal  ScalingReplicaSet  1m   deployment-controller  Scaled up replica set kubernetes-bootcamp-765bf4c7b4 to 1

$ kubectl get pods
NAME                                   READY   STATUS    RESTARTS   AGE
kubernetes-bootcamp-765bf4c7b4-mfhrm   1/1     Running   0          1m

$ kubectl describe pods
Name:         kubernetes-bootcamp-765bf4c7b4-mfhrm
Namespace:    default
Priority:     0
Node:         minikube/172.17.0.115    # 从属Node
Start Time:   Tue, 15 Dec 2020 08:43:46 +0000
Labels:       pod-template-hash=765bf4c7b4
              run=kubernetes-bootcamp
Annotations:  <none>
Status:       Running
IP:           172.18.0.4               # IP
IPs:
  IP:           172.18.0.4
Controlled By:  ReplicaSet/kubernetes-bootcamp-765bf4c7b4
Containers:                            # 容器列表
  kubernetes-bootcamp:
    Container ID:   docker://6acccaf5534b86fe516c809d421b4967b4049c7dfa2ad31719fb9f01213d5bf4
    Image:          gcr.io/google-samples/kubernetes-bootcamp:v1
    Image ID:       docker-pullable://jocatalin/kubernetes-bootcamp@sha256:0d6b8ee63bb57c5f5b6156f446b3bc3b3c143d233037f3a2f00e279c8fcc64af
    Port:           8080/TCP
    Host Port:      0/TCP
    State:          Running
      Started:      Tue, 15 Dec 2020 08:43:48 +0000
    Ready:          True
    Restart Count:  0
    Environment:    <none>
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from default-token-4w4rq (ro)
Conditions:
  Type              Status
  Initialized       True
  Ready             True
  ContainersReady   True
  PodScheduled      True
Volumes:
  default-token-4w4rq:
    Type:        Secret (a volume populated by a Secret)
    SecretName:  default-token-4w4rq
    Optional:    false
QoS Class:       BestEffort
Node-Selectors:  <none>
Tolerations:     node.kubernetes.io/not-ready:NoExecute for 300s
                 node.kubernetes.io/unreachable:NoExecute for 300s
Events:
  Type     Reason            Age                    From               Message
  ----     ------            ----                   ----               -------
  Warning  FailedScheduling  1m29s (x2 over 1m30s)  default-scheduler  0/1 nodes areavailable: 1 node(s) had taints that the pod didn't tolerate.
  Normal   Scheduled         1m24s                  default-scheduler  Successfully assigned default/kubernetes-bootcamp-765bf4c7b4-mfhrm to minikube
  Normal   Pulled            1m22s                  kubelet, minikube  Container image "gcr.io/google-samples/kubernetes-bootcamp:v1" already present on machine
  Normal   Created           1m22s                  kubelet, minikube  Created container kubernetes-bootcamp
  Normal   Started           1m22s                  kubelet, minikube  Started container kubernetes-bootcamp

# * 使用proxy
# 在另一个terminal执行
$ kubectl proxy
Starting to serve on 127.0.0.1:8001

$ curl http://localhost:8001/version
{
  "major": "1",
  "minor": "17",
  "gitVersion": "v1.17.3",
  "gitCommit": "06ad960bfd03b39c8310aaf92d1e7c12ce618213",
  "gitTreeState": "clean",
  "buildDate": "2020-02-11T18:07:13Z",
  "goVersion": "go1.13.6",
  "compiler": "gc",
  "platform": "linux/amd64"
}
# 相当于请求Master
$ curl https://172.17.0.115:8443/version
{
  "major": "1",
  "minor": "17",
  "gitVersion": "v1.17.3",
  "gitCommit": "06ad960bfd03b39c8310aaf92d1e7c12ce618213",
  "gitTreeState": "clean",
  "buildDate": "2020-02-11T18:07:13Z",
  "goVersion": "go1.13.6",
  "compiler": "gc",
  "platform": "linux/amd64"
}

$ export POD_NAME=kubernetes-bootcamp-765bf4c7b4-mfhrm
$ curl http://localhost:8001/api/v1/namespaces/default/pods/$POD_NAME/proxy/
Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-765bf4c7b4-mfhrm | v=1
# 但是
$ curl https://172.17.0.115:8443/api/v1/namespaces/default/pods/$POD_NAME/proxy/
# 访问被拒绝,集群内部资源必须通过proxy访问

# 3. 在Pod上执行命令
# 执行env命令
$ kubectl exec $POD_NAME env
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HOSTNAME=kubernetes-bootcamp-765bf4c7b4-mfhrm
KUBERNETES_PORT_443_TCP_ADDR=10.96.0.1
KUBERNETES_SERVICE_HOST=10.96.0.1
KUBERNETES_SERVICE_PORT=443
KUBERNETES_SERVICE_PORT_HTTPS=443
KUBERNETES_PORT=tcp://10.96.0.1:443
KUBERNETES_PORT_443_TCP=tcp://10.96.0.1:443
KUBERNETES_PORT_443_TCP_PROTO=tcp
KUBERNETES_PORT_443_TCP_PORT=443
NPM_CONFIG_LOGLEVEL=info
NODE_VERSION=6.3.1
HOME=/root
# 启动bash
$ kubectl exec -ti $POD_NAME bash
root@kubernetes-bootcamp-765bf4c7b4-mfhrm:/$ ls
bin   core  etc   lib    media  opt   root  sbin       srv  tmp  var
boot  dev   home  lib64  mnt    proc  run   server.js  sys  usr
root@kubernetes-bootcamp-765bf4c7b4-mfhrm:/$ curl localhost:8080
Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-765bf4c7b4-mfhrm | v=1
root@kubernetes-bootcamp-765bf4c7b4-mfhrm:/$ exit
exit

# 查看Service信息
$ kubectl get service
NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
# 集群启动时的默认service
kubernetes   ClusterIP   10.96.0.1    <none>        443/TCP   1m
# 4. 暴露Deployment为Service
$ kubectl expose deployment/kubernetes-bootcamp --type="NodePort" --port 8080
service/kubernetes-bootcamp exposed
# 查看Service信息
$ kubectl get service
NAME                  TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)          AGE
kubernetes            ClusterIP   10.96.0.1        <none>        443/TCP          1m
# NodePort类型,将Pod的8080端口映射到Node的30207端口
kubernetes-bootcamp   NodePort    10.108.159.233   <none>        8080:30207/TCP   1m

$ kubectl describe services/kubernetes-bootcamp
Name:                     kubernetes-bootcamp
Namespace:                default
Labels:                   run=kubernetes-bootcamp
Annotations:              <none>
Selector:                 run=kubernetes-bootcamp
Type:                     NodePort
IP:                       10.108.159.233
Port:                     <unset>  8080/TCP      # Service端口
TargetPort:               8080/TCP               # Pod端口
NodePort:                 <unset>  30207/TCP     # Node端口
Endpoints:                172.18.0.4:8080        # 唯一一个Pod的IP+Port
Session Affinity:         None
External Traffic Policy:  Cluster
Events:                   <none>

# 5. 请求该Service
$ export NODE_PORT=30207
$ curl $(minikube ip):$NODE_PORT
Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-765bf4c7b4-mfhrm | v=1
# Service下只有一个Pod,对Service的请求必定被转发到该Pod

# 通过selector查找Pod
$ kubectl get pods -l run=kubernetes-bootcamp
NAME                                   READY   STATUS    RESTARTS   AGE
kubernetes-bootcamp-765bf4c7b4-mfhrm   1/1     Running   0          1m

# 查看Pod的标签
$ kubectl get pod --show-labels
NAME                                   READY   STATUS    RESTARTS   AGE   LABELS
kubernetes-bootcamp-765bf4c7b4-mfhrm   1/1     Running   0          1m   pod-template-hash=765bf4c7b4,run=kubernetes-bootcamp
# 6. 为Pod增加标签
$ kubectl label pod $POD_NAME app=v1
pod/kubernetes-bootcamp-765bf4c7b4-mfhrm labeled
$ kubectl get pod --show-labels
NAME                                   READY   STATUS    RESTARTS   AGE   LABELS
kubernetes-bootcamp-765bf4c7b4-mfhrm   1/1     Running   0          1m   app=v1,pod-template-hash=765bf4c7b4,run=kubernetes-bootcamp

# 7. 删除Service
$ kubectl delete service -l run=kubernetes-bootcamp
service "kubernetes-bootcamp" deleted
# 连接被拒绝
$ curl $(minikube ip):$NODE_PORT
curl: (7) Failed to connect to 172.17.0.59 port 31834: Connection refused
# 但Pod仍然在运行
$ kubectl exec -ti $POD_NAME curl localhost:8080
Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-765bf4c7b4-mfhrm | v=1

# 查看Deployment信息
$ kubectl get deployment
NAME                  READY   UP-TO-DATE   AVAILABLE   AGE
kubernetes-bootcamp   1/1     1            1           1m
#                   现有/需要的  最新镜像的  可以提供服务的  Pod数
# 查看Deployment创建的ReplicaSet信息
$ kubectl get rs
NAME                             DESIRED   CURRENT   READY   AGE
kubernetes-bootcamp-765bf4c7b4   1         1         1       1m
# 8. 将Deployment扩容到4个replica
$ kubectl scale deployments/kubernetes-bootcamp --replicas=4
deployment.apps/kubernetes-bootcamp scaled
$ kubectl get deployments
NAME                  READY   UP-TO-DATE   AVAILABLE   AGE
kubernetes-bootcamp   4/4     4            4           1m
# 查看扩容后的信息
$ kubectl get pods -o wide
NAME                                   READY   STATUS    RESTARTS   AGE     IP    NODE       NOMINATED NODE   READINESS GATES
kubernetes-bootcamp-765bf4c7b4-mfhrm   1/1     Running   0          3m35s   172.18.0.4   minikube   <none>           <none>   # 起初的1个Pod
kubernetes-bootcamp-765bf4c7b4-kmss4   1/1     Running   0          2m12s   172.18.0.8   minikube   <none>           <none>   # 新创建的Pod
kubernetes-bootcamp-765bf4c7b4-m62gl   1/1     Running   0          2m12s   172.18.0.9   minikube   <none>           <none>
kubernetes-bootcamp-765bf4c7b4-t26sf   1/1     Running   0          2m13s   172.18.0.7   minikube   <none>           <none>
$ kubectl describe deployments/kubernetes-bootcamp
Name:                   kubernetes-bootcamp
Namespace:              default
CreationTimestamp:      Tue, 15 Dec 2020 12:40:22 +0000
Labels:                 run=kubernetes-bootcamp
Annotations:            deployment.kubernetes.io/revision: 1
Selector:               run=kubernetes-bootcamp
Replicas:               4 desired | 4 updated | 4 total | 4 available | 0 unavailable
StrategyType:           RollingUpdate
MinReadySeconds:        0
RollingUpdateStrategy:  25% max unavailable, 25% max surge
Pod Template:
  Labels:  run=kubernetes-bootcamp
  Containers:
   kubernetes-bootcamp:
    Image:        gcr.io/google-samples/kubernetes-bootcamp:v1
    Port:         8080/TCP
    Host Port:    0/TCP
    Environment:  <none>
    Mounts:       <none>
  Volumes:        <none>
Conditions:
  Type           Status  Reason
  ----           ------  ------
  Progressing    True    NewReplicaSetAvailable
  Available      True    MinimumReplicasAvailable
OldReplicaSets:  <none>
NewReplicaSet:   kubernetes-bootcamp-765bf4c7b4 (4/4 replicas created)
Events:
  Type    Reason             Age    From                   Message
  ----    ------             ----   ----                   -------
  Normal  ScalingReplicaSet  7m19s  deployment-controller  Scaled up replica set kubernetes-bootcamp-765bf4c7b4 to 1
  Normal  ScalingReplicaSet  5m56s  deployment-controller  Scaled up replica set kubernetes-bootcamp-765bf4c7b4 to 4   # 扩容事件
$ kubectl describe services/kubernetes-bootcamp
Name:                     kubernetes-bootcamp
Namespace:                default
Labels:                   run=kubernetes-bootcamp
Annotations:              <none>
Selector:                 run=kubernetes-bootcamp
Type:                     NodePort
IP:                       10.108.159.233
Port:                     <unset>  8080/TCP
TargetPort:               8080/TCP
NodePort:                 <unset>  30207/TCP
Endpoints:                172.18.0.4:8080,172.18.0.7:8080,172.18.0.8:8080 + 1 more...      # 现在有4个endpoint
Session Affinity:         None
External Traffic Policy:  Cluster
Events:                   <none>

# 9. 再次请求该Service
$ export NODE_PORT=30207
$ curl $(minikube ip):$NODE_PORT
Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-765bf4c7b4-kmss4 | v=1
$ curl $(minikube ip):$NODE_PORT
Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-765bf4c7b4-m62gl | v=1      # 负载均匀
$ curl $(minikube ip):$NODE_PORT
Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-765bf4c7b4-t26sf | v=1

# 10. 更新Deployment的镜像
$ kubectl set image deployments/kubernetes-bootcamp kubernetes-bootcamp=jocatalin/kubernetes-bootcamp:v2
deployment.apps/kubernetes-bootcamp image updated
# 原Pod逐渐关闭,新Pod逐渐创建
$ kubectl get pods
NAME                                   READY   STATUS              RESTARTS   AGE
kubernetes-bootcamp-756f66956c-mfhrm   1/1     Running             0          2m
kubernetes-bootcamp-756f66956c-kmss4   1/1     Running             0          1m
kubernetes-bootcamp-756f66956c-m62gl   1/1     Terminating         0          1m
kubernetes-bootcamp-756f66956c-t26sf   1/1     Running             0          1m
kubernetes-bootcamp-7d6f8694b6-hj4cw   0/1     ContainerCreating   0          1s
kubernetes-bootcamp-7d6f8694b6-jb9ds   0/1     ContainerCreating   0          1s
# 新Pod全部就绪之前,保留1个原Pod
$ kubectl get pods
NAME                                   READY   STATUS              RESTARTS   AGE
kubernetes-bootcamp-756f66956c-mfhrm   1/1     Terminating         0          2m
kubernetes-bootcamp-756f66956c-kmss4   1/1     Terminating         0          1m
kubernetes-bootcamp-756f66956c-m62gl   1/1     Terminating         0          1m
kubernetes-bootcamp-756f66956c-t26sf   1/1     Running             0          1m
kubernetes-bootcamp-7d6f8694b6-gt8hw   0/1     ContainerCreating   0          0s
kubernetes-bootcamp-7d6f8694b6-hj4cw   1/1     Running             0          3s
kubernetes-bootcamp-7d6f8694b6-jb9ds   1/1     Running             0          3s
kubernetes-bootcamp-7d6f8694b6-z9qkz   0/1     ContainerCreating   0          0s
# 更新完毕
$ kubectl get pods
NAME                                   READY   STATUS        RESTARTS   AGE
kubernetes-bootcamp-756f66956c-mfhrm   1/1     Terminating   0           2m
kubernetes-bootcamp-756f66956c-kmss4   1/1     Terminating   0           1m
kubernetes-bootcamp-756f66956c-m62gl   1/1     Terminating   0           1m
kubernetes-bootcamp-756f66956c-t26sf   1/1     Terminating   0           1m
kubernetes-bootcamp-7d6f8694b6-gt8hw   1/1     Running       0          3s
kubernetes-bootcamp-7d6f8694b6-hj4cw   1/1     Running       0          6s
kubernetes-bootcamp-7d6f8694b6-jb9ds   1/1     Running       0          6s
kubernetes-bootcamp-7d6f8694b6-z9qkz   1/1     Running       0          3s
$ kubectl describe services/kubernetes-bootcamp
Name:                     kubernetes-bootcamp
Namespace:                default
Labels:                   run=kubernetes-bootcamp
Annotations:              <none>
Selector:                 run=kubernetes-bootcamp
Type:                     NodePort
IP:                       10.108.159.233
Port:                     <unset>  8080/TCP
TargetPort:               8080/TCP
NodePort:                 <unset>  30207/TCP
# endpoint全部更新
Endpoints:                172.18.0.10:8080,172.18.0.11:8080,172.18.0.12:8080 + 1 more...
Session Affinity:         None
External Traffic Policy:  Cluster
Events:                   <none>
$ curl $(minikube ip):$NODE_PORT
Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-7d6f8694b6-hj4cw | v=2
# 查看最近的滚动更新
$ kubectl rollout history deployments/kubernetes-bootcamp
deployment.apps/kubernetes-bootcamp
REVISION  CHANGE-CAUSE
1         <none>
2         <none>
$ kubectl rollout history deployments/kubernetes-bootcamp --revision=1
deployment.apps/kubernetes-bootcamp with revision #1
Pod Template:
  Labels:       pod-template-hash=765bf4c7b4
        run=kubernetes-bootcamp
  Containers:
   kubernetes-bootcamp:   # 更新前镜像
    Image:      gcr.io/google-samples/kubernetes-bootcamp:v1
    Port:       8080/TCP
    Host Port:  0/TCP
    Environment:        <none>
    Mounts:     <none>
  Volumes:      <none>
$ kubectl rollout history deployments/kubernetes-bootcamp --revision=2
deployment.apps/kubernetes-bootcamp with revision #2
Pod Template:
  Labels:       pod-template-hash=7d6f8694b6
        run=kubernetes-bootcamp
  Containers:
   kubernetes-bootcamp:   # 更新后镜像
    Image:      jocatalin/kubernetes-bootcamp:v2
    Port:       8080/TCP
    Host Port:  0/TCP
    Environment:        <none>
    Mounts:     <none>
  Volumes:      <none>
  
# 11. 再次更新Deployment的镜像,但新镜像不存在
$ kubectl set image deployments/kubernetes-bootcamp kubernetes-bootcamp=gcr.io/google-samples/kubernetes-bootcamp:v10
deployment.apps/kubernetes-bootcamp image updated
$ kubectl get pods
NAME                                   READY   STATUS             RESTARTS   AGE
kubernetes-bootcamp-7d6f8694b6-gt8hw   1/1     Running            0          1m
kubernetes-bootcamp-7d6f8694b6-hj4cw   1/1     Running            0          1m
kubernetes-bootcamp-7d6f8694b6-z9qkz   1/1     Running            0          1m
# 镜像拉取失败,无法创建新Pod;原Pod仅删除了1个
kubernetes-bootcamp-886577c5d-kp6z7    0/1     ErrImagePull       0          33s
kubernetes-bootcamp-886577c5d-wmhl5    0/1     ImagePullBackOff   0          32s
# 查看最近的滚动更新
$ kubectl rollout history deployments/kubernetes-bootcamp --revision=3
deployment.apps/kubernetes-bootcamp with revision #3
Pod Template:
  Labels:       pod-template-hash=886577c5d
        run=kubernetes-bootcamp
  Containers:
   kubernetes-bootcamp:
    Image:      gcr.io/google-samples/kubernetes-bootcamp:v10
    Port:       8080/TCP
    Host Port:  0/TCP
    Environment:        <none>
    Mounts:     <none>
  Volumes:      <none>
# 回滚到revision=2
$ kubectl rollout undo deployments/kubernetes-bootcamp
deployment.apps/kubernetes-bootcamp rolled back
$ kubectl get pods
NAME                                   READY   STATUS              RESTARTS   AGE
# 停用新Pod,重新创建上个版本的Pod
kubernetes-bootcamp-7d6f8694b6-hb9nm   0/1     ContainerCreating   0          0s
kubernetes-bootcamp-7d6f8694b6-gt8hw   1/1     Running             0          1m
kubernetes-bootcamp-7d6f8694b6-hj4cw   1/1     Running             0          1m
kubernetes-bootcamp-7d6f8694b6-z9qkz   1/1     Running             0          1m
kubernetes-bootcamp-886577c5d-kp6z7    0/1     Terminating         0          1m
kubernetes-bootcamp-886577c5d-wmhl5    0/1     Terminating         0          1m
$ kubectl get pods
NAME                                   READY   STATUS              RESTARTS   AGE
kubernetes-bootcamp-7d6f8694b6-hb9nm   1/1     Running             0          1s
kubernetes-bootcamp-7d6f8694b6-gt8hw   1/1     Running             0          1m
kubernetes-bootcamp-7d6f8694b6-hj4cw   1/1     Running             0          1m
kubernetes-bootcamp-7d6f8694b6-z9qkz   1/1     Running             0          1m

```

