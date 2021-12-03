

# 基本命令

## create

使用YAML或JSON文件创建一个资源。

```shell
# Create a pod using the data in pod.yaml
$ kubectl create -f ./pod.yaml
# Flags
# --filename -f  Filename, directory or URL to files to use to create the resource
```

下列诸命令使用命令行参数，而非配置文件创建资源，基本上用于测试过程。

### deployment

```shell
# Create a deployment named my-dep that runs the nginx image with 3 replicas
$ kubectl create deployment my-dep --image=nginx --replicas=3
# Flags
# --image        Image names to run.
# --port         Port that container exposes.
# --replicas -r  Number of replicas to create. Default is 1.
```

### namespace

```shell
# Create a new namespace named my-namespace
$ kubectl create namespace my-namespace
```

## get

获取各种资源的信息。

```shell
# List all pods in ps output format
$ kubectl get pod
# Flags
# --selector -l  Selector to filter on (e.g. -l key1=value1,key2=value2)
# --show-labels	 Show all labels as the last column (default hide)
```

```shell
# 获取所有节点信息
$ kubectl get node
NAME           STATUS   ROLES    AGE     VERSION
controlplane   Ready    master   2m18s   v1.18.0  # Master
node01         Ready    <none>   106s    v1.18.0  # Node
```

## run

创建一个Pod，运行指定镜像。

```shell
# Start a nginx pod
$ kubectl run nginx --image=nginx
```

## expose

将资源暴露为Service。

通过名称指定Deployment, Service, Replica set, RC 或 Pod，然后使用该资源的selector作为selector创建新的 Service。Deployment或Replica set只有当其选择器可转换为Service支持的选择器时，即当选择器仅包含matchLabels组件时才会作为暴露新的Service。

如果没有通过`--port`指定端口，并且暴露的资源有多个端口，那么所有的端口都会被Service复用。

```shell
# Create a service for an nginx deployment, which serves on port 80 and connects to the containers on port 8000
$ kubectl expose deployment nginx --port=80 --target-port=8000
# Flags
# --port         Port that the service should serve on. Copied from the resource being exposed, if unspecified
# --type         Type for this service: ClusterIP, NodePort, LoadBalancer, or ExternalName. Default is 'ClusterIP'.
# --target-port  Port on the container that the service should direct traffic to
```

## delete

根据配置文件、资源名称或资源Label删除资源。

```shell
# Delete pods using the type and name specified in pod.yaml
# 根据某配置文件创建的资源也会被该配置文件删除
$ kubectl delete -f ./pod.yaml

# Delete pods and services with name same as 'baz' or 'foo'
$ kubectl delete pod,service baz foo

# Delete pods and services with label name=myLabel.
$ kubectl delete pods,services -l name=myLabel
```

# 应用管理

## apply

应用YAML或JSON文件创建一个资源。与`create`功能相同，但一般使用`apply`。

```shell
# Apply the configuration in pod.yaml to a pod
$ kubectl apply -f ./pod.yaml
# Flags
# --filename -f  Filename, directory or URL to files to use to create the resource
```

## label

更新资源的label。

```shell
# Add label app=v1 for pod 'foo'
$ kubectl label pods foo app=v1

# Update label app=v2 for pod 'foo', overwriting any existing value
$ kubectl label --overwrite pods foo app=v2

# Remove label app=* for pod 'foo'
$ kubectl label pods foo app-
# Flag
# --all          Select all resources, including uninitialized ones, in the namespace of the specified resource types
# --filename -f  Filename, directory or URL to files to use to create the resource
# --overwrite    allow labels to be overwritten
# --selector -l  Selector to filter on, not including uninitialized ones (e.g. -l key1=value1,key2=value2)

```

## rollout

管理资源的滚动更新。

### history

查看滚动更新的最近修订。

```shell
# View the rollout history of a deployment
$ kubectl rollout history deployment/abc

# View the details of daemonset revision 3
$ kubectl rollout history daemonset/abc --revision=3
# Flag
# --filename -f  Filename, directory or URL to files to use to create the resource
# --revision     See the details, including podTemplate of the revision specified
```

### status

展示滚动更新的状态。默认情况下会监视最近一次滚动更新的状态，如果你不想等待滚动更新结束，可以使用`--watch=false`。

```shell
# 更新Deployment的镜像
$ kubectl set image deployments/kubernetes-bootcamp kubernetes-bootcamp=jocatalin/kubernetes-bootcamp:v2
deployment.apps/kubernetes-bootcamp image updated
# 监视滚动更新的状态
$ kubectl rollout status deployments/kubernetes-bootcamp
Waiting for deployment "kubernetes-bootcamp" rollout to finish: 2 out of 4 new replicas have been updated...
Waiting for deployment "kubernetes-bootcamp" rollout to finish: 2 out of 4 new replicas have been updated...
Waiting for deployment "kubernetes-bootcamp" rollout to finish: 2 out of 4 new replicas have been updated...
Waiting for deployment "kubernetes-bootcamp" rollout to finish: 2 out of 4 new replicas have been updated...
Waiting for deployment "kubernetes-bootcamp" rollout to finish: 2 out of 4 new replicas have been updated...
Waiting for deployment "kubernetes-bootcamp" rollout to finish: 3 out of 4 new replicas have been updated...
Waiting for deployment "kubernetes-bootcamp" rollout to finish: 1 old replicas are pending termination...
Waiting for deployment "kubernetes-bootcamp" rollout to finish: 1 old replicas are pending termination...
Waiting for deployment "kubernetes-bootcamp" rollout to finish: 1 old replicas are pending termination...
Waiting for deployment "kubernetes-bootcamp" rollout to finish: 1 old replicas are pending termination...
deployment "kubernetes-bootcamp" successfully rolled out
```

### undo

回滚到最近一次的滚动更新。

## scale

为Deployment, ReplicaSet设定一个新的大小。

```shell
# Scale a replicaset named 'foo' to 3
$ kubectl scale --replicas=3 rs/foo

# Scale a resource identified by type and name specified in "foo.yaml" to 3.
$ kubectl scale --replicas=3 -f foo.yaml
```

## set

更改现有的应用资源。

```shell
# 更改deployment的image以进行滚动升级
$ kubectl set image deployments/kubernetes-bootcamp kubernetes-bootcamp=jocatalin/kubernetes-bootcamp:v2

```

# 检查应用

## describe

展示资源的详细信息。

```shell
# 获取所有Pod详细信息
$ kubectl describe pod
Name:         kubernetes-bootcamp-765bf4c7b4-p4td5       # 名称
Namespace:    default                                    # namespace
Priority:     0
Node:         minikube/172.17.0.67                       # 从属的Node
Start Time:   Tue, 15 Dec 2020 06:09:46 +0000
Labels:       pod-template-hash=765bf4c7b4               # label
              run=kubernetes-bootcamp
Annotations:  <none>
Status:       Running
IP:           172.18.0.5                                 # IP
IPs:
  IP:           172.18.0.5
Controlled By:  ReplicaSet/kubernetes-bootcamp-765bf4c7b4  # 由deployment控制
Containers:                                              # Pod下的所有容器
  kubernetes-bootcamp:                                   # 容器名
    Container ID:   docker://e597f15e4ac1949e3f98cdb0e75f3058d0b9c9f4a6a1c8b454faaa8170dd3ec6  # 容器ID
    Image:          gcr.io/google-samples/kubernetes-bootcamp:v1           # 容器镜像
    Image ID:       docker-pullable://jocatalin/kubernetes-bootcamp@sha256:0d6b8ee63bb57c5f5b6156f446b3bc3b3c143d233037f3a2f00e279c8fcc64af
    Port:           8080/TCP                             # 容器暴露端口
    Host Port:      0/TCP
    State:          Running
      Started:      Tue, 15 Dec 2020 06:09:52 +0000
    Ready:          True
    Restart Count:  0
    Environment:    <none>
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from default-token-vr7ls (ro)
Conditions:
  Type              Status
  Initialized       True
  Ready             True
  ContainersReady   True
  PodScheduled      True
Volumes:
  default-token-vr7ls:
    Type:        Secret (a volume populated by a Secret)
    SecretName:  default-token-vr7ls
    Optional:    false
QoS Class:       BestEffort
Node-Selectors:  <none>
Tolerations:     node.kubernetes.io/not-ready:NoExecute for 300s
                 node.kubernetes.io/unreachable:NoExecute for 300s
Events:
  Type     Reason            Age                From               Message
  ----     ------            ----               ----               -------
  Warning  FailedScheduling  42s (x4 over 48s)  default-scheduler  0/1 nodes are available: 1 node(s) had taints that the pod didn't tolerate.
  Normal   Scheduled         37s                default-scheduler  Successfully assigned default/kubernetes-bootcamp-765bf4c7b4-p4td5 to minikube
  Normal   Pulled            33s                kubelet, minikube  Container image "gcr.io/google-samples/kubernetes-bootcamp:v1" already present on machine
  Normal   Created           32s                kubelet, minikube  Created containerkubernetes-bootcamp
  Normal   Started           31s                kubelet, minikube  Started containerkubernetes-bootcamp
```

```shell
# 获取Node详细信息
$ kubectl describe node node01
Name:               node01                                # 名称
Roles:              <none>                                # 非Master
Labels:             beta.kubernetes.io/arch=amd64         # 标签
                    beta.kubernetes.io/os=linux
                    kubernetes.io/arch=amd64
                    kubernetes.io/hostname=node01
                    kubernetes.io/os=linux
Annotations:        flannel.alpha.coreos.com/backend-data: null
                    flannel.alpha.coreos.com/backend-type: host-gw
                    flannel.alpha.coreos.com/kube-subnet-manager: true
                    flannel.alpha.coreos.com/public-ip: 172.17.0.82
                    kubeadm.alpha.kubernetes.io/cri-socket: /var/run/dockershim.sock
                    node.alpha.kubernetes.io/ttl: 0
                    volumes.kubernetes.io/controller-managed-attach-detach: true
CreationTimestamp:  Mon, 14 Dec 2020 08:12:55 +0000       # 创建时间
Taints:             <none>
Unschedulable:      false
Lease:
  HolderIdentity:  node01
  AcquireTime:     <unset>
  RenewTime:       Mon, 14 Dec 2020 08:15:37 +0000
Conditions:        # 进行网络、内存、磁盘、进程检查,检查完毕后Ready状态为True,表示Node一切正常,可以在其上创建新的Pod
  Type                 Status  LastHeartbeatTime                 LastTransitionTime                Reason                       Message
  ----                 ------  -----------------                 ------------------                ------                       -------
  NetworkUnavailable   False   Mon, 14 Dec 2020 08:13:16 +0000   Mon, 14 Dec 2020 08:13:16 +0000   FlannelIsUp                  Flannel is running on this node
  MemoryPressure       False   Mon, 14 Dec 2020 08:13:56 +0000   Mon, 14 Dec 2020 08:12:56 +0000   KubeletHasSufficientMemory   kubelet has sufficient memory available
  DiskPressure         False   Mon, 14 Dec 2020 08:13:56 +0000   Mon, 14 Dec 2020 08:12:56 +0000   KubeletHasNoDiskPressure     kubelet has no disk pressure
  PIDPressure          False   Mon, 14 Dec 2020 08:13:56 +0000   Mon, 14 Dec 2020 08:12:56 +0000   KubeletHasSufficientPID      kubelet has sufficient PID available
  Ready                True    Mon, 14 Dec 2020 08:13:56 +0000   Mon, 14 Dec 2020 08:13:16 +0000   KubeletReady                 kubelet is posting ready status. AppArmor enabled
Addresses:                                                 # 主机地址
  InternalIP:  172.17.0.82
  Hostname:    node01
Capacity:                                                  # 资源总量
  cpu:                2
  ephemeral-storage:  199545168Ki
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             4039104Ki
  pods:               110
Allocatable:                                               # 可分配资源量
  cpu:                2
  ephemeral-storage:  183900826525
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             3936704Ki
  pods:               110
System Info:                                               # 系统信息
  Machine ID:                 28b20a54088bb20c134f68c75fd71e54
  System UUID:                28b20a54088bb20c134f68c75fd71e54
  Boot ID:                    d7aeed47-afaa-40ff-ac1c-d9938acde9f2
  Kernel Version:             4.15.0-122-generic
  OS Image:                   Ubuntu 18.04.5 LTS
  Operating System:           linux
  Architecture:               amd64
  Container Runtime Version:  docker://19.3.13
  Kubelet Version:            v1.18.0
  Kube-Proxy Version:         v1.18.0
PodCIDR:                      10.244.1.0/24
PodCIDRs:                     10.244.1.0/24
Non-terminated Pods:          (4 in total)                 # 正在运行的Pod信息
  Namespace                   Name                                       CPU Requests  CPU Limits  Memory Requests  Memory Limits  AGE
  ---------                   ----                                       ------------  ----------  ---------------  -------------  ---
  kube-system                 katacoda-cloud-provider-58f89f7d9-g7sjv    200m (10%)    0 (0%)      0 (0%)           0 (0%)         2m53s
  kube-system                 kube-flannel-ds-amd64-tz79g                100m (5%)     100m (5%)   50Mi (1%)        50Mi (1%)      2m43s
  kube-system                 kube-keepalived-vip-cxnjj                  0 (0%)        0 (0%)      0 (0%)           0 (0%)         2m23s
  kube-system                 kube-proxy-54kns                           0 (0%)        0 (0%)      0 (0%)           0 (0%)         2m43s
Allocated resources:
  (Total limits may be over 100 percent, i.e., overcommitted.)
  Resource           Requests    Limits
  --------           --------    ------
  cpu                300m (15%)  100m (5%)
  memory             50Mi (1%)   50Mi (1%)
  ephemeral-storage  0 (0%)      0 (0%)
  hugepages-1Gi      0 (0%)      0 (0%)
  hugepages-2Mi      0 (0%)      0 (0%)
Events:
  Type    Reason                   Age                    From                Message
  ----    ------                   ----                   ----                -------
  Normal  Starting                 2m44s                  kubelet, node01     Starting kubelet.
  Normal  NodeHasSufficientMemory  2m43s (x2 over 2m43s)  kubelet, node01     Node node01 status is now: NodeHasSufficientMemory
  Normal  NodeHasNoDiskPressure    2m43s (x2 over 2m43s)  kubelet, node01     Node node01 status is now: NodeHasNoDiskPressure
  Normal  NodeHasSufficientPID     2m43s (x2 over 2m43s)  kubelet, node01     Node node01 status is now: NodeHasSufficientPID
  Normal  NodeAllocatableEnforced  2m43s                  kubelet, node01     Updated Node Allocatable limit across pods
  Normal  Starting                 2m30s                  kube-proxy, node01  Starting kube-proxy.
  Normal  NodeReady                2m23s                  kubelet, node01     Node node01 status is now: NodeReady
```

## exec

在容器上执行命令。

```shell
# 在容器上执行命令env
$ kubectl exec mypod env

# 在容器上启动bash
$ kubectl exec -it mypod bash
root@kubernetes-bootcamp-765bf4c7b4-p4td5:/#
# Flag
# --container -c  Container name. If omitted, first container in the pod will be chosen
```

## proxy

在本地端口上创建到Kubernetes apiserver的代理服务器。

