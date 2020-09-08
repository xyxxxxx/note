> 参考[Kubernetes 初体验](https://www.qikqiak.com/k8s-book/docs/14.Kubernetes%E5%88%9D%E4%BD%93%E9%AA%8C.html)，[学习Kubernetes基础知识](https://kubernetes.io/zh/docs/tutorials/kubernetes-basics/)



# 什么是Kubernetes?

> 参考[Kubernetes 是什么？](https://kubernetes.io/zh/docs/concepts/overview/what-is-kubernetes/)

Kubernetes 是 Google 团队发起的一个开源项目，它的目标是管理跨多个主机的容器，用于自动部署、扩展和管理容器化的应用程序，主要实现语言为 Go 语言。





# 架构

![](https://d33wubrfki0l68.cloudfront.net/7016517375d10c702489167e704dcb99e570df85/7bb53/images/docs/components-of-kubernetes.png)



## 集群

一个 Kubernetes 集群由一组被称作节点的机器组成。这些节点上运行 Kubernetes 所管理的容器化应用。集群具有至少一个工作节点和至少一个主节点。



## Master

![k8s cluster](https://www.qikqiak.com/k8s-book/docs/images/k8s-cluster.png)

Master（主节点）是 Kubernetes 集群的控制节点，负责整个集群的管理和控制，包含以下组件：

+ kube-apiserver：集群控制的入口，提供 HTTP REST 服务，与 Node 以及 kubeclt 通信
+ kube-controller-manager：集群中所有资源对象的自动化控制中心
  + Node Controller（节点控制器）: 负责在节点出现故障时进行通知和响应。
  + Replication Controller（副本控制器）: 负责为系统中的每个副本控制器对象维护正确数量的 Pod。
  + Endpoints Controller（端点控制器）: 填充端点(Endpoints)对象(即加入 Service 与 Pod)。
  + Service Account & Token Controllers（服务帐户和令牌控制器）: 为新的命名空间创建默认帐户和 API 访问令牌.
+ kube-scheduler：负责 Pod 的调度，即监视新创建的 Pod 并为其分配 Node
+ etcd：保存 Kubernetes 所有集群数据的后台数据库
+ cloud-controller-manager：云控制器管理器是 1.8 的 alpha 特性。在未来发布的版本中，这是将 Kubernetes 与任何其他云集成的最佳方式。



## Pod

![](https://d33wubrfki0l68.cloudfront.net/fe03f68d8ede9815184852ca2a4fd30325e5d15a/98064/docs/tutorials/kubernetes-basics/public/images/module_03_pods.svg)

Pod 是一组紧密关联的容器集合，它们共享 PID、IPC、存储、网络 和 UTS namespace，是Kubernetes 调度的基本单元。Pod 的设计理念是支持多个容器在一个 Pod 中共享网络和文件系统，可以通过进程间通信和文件共享这种简单高效的方式组合完成服务。Pod 中的容器始终位于同一位置并且共同调度，并在同一工作节点上的共享上下文中运行。

一个 Pod 表示某个应用的一个实例。



## Node

![](https://d33wubrfki0l68.cloudfront.net/5cb72d407cbe2755e581b6de757e0d81760d5b86/a9df9/docs/tutorials/kubernetes-basics/public/images/module_03_nodes.svg)

Node（工作节点）是 Kubernetes 中的参与计算的机器，可以是虚拟机或物理计算机。每个工作节点由主节点管理。工作节点可以有多个 pod ，Kubernetes 主节点会自动处理在群集中的工作节点上调度 pod 。 主节点的自动调度考量了每个工作节点上的可用资源。

Node 包含以下组件：

+ kubelet，负责与主节点的通信，并且管理 Pod 的创建、启动、监控、重启、销毁等工作
+ container runtime（如 Docker），负责运行容器的软件
+ kube-proxy，实现 Service 的服务发现和负载均衡



## Label

Label（标签）是识别 Kubernetes 对象的标签，以 key/value 的方式附加到对象上（key最长不能超过63字节，value 可以为空，也可以是不超过253字节的字符串）。 Label 不提供唯一性，并且实际上经常是很多对象（如Pods）都使用相同的 label 来标志具体的应用。 Label 定义好后其他对象可以使用 Label Selector 来选择一组相同 label 的对象（比如Service 用 label 来选择一组 Pod）。Label Selector支持以下几种方式：

+ 等式，如`app=nginx`和`env!=production`
+ 集合，如`env in (production, qa)`
+ 多个label（它们之间是AND关系），如`app=nginx,env=test`



## Namespace

Namespace 是对一组资源和对象的抽象集合，比如可以用来将系统内部的对象划分为不同的项目组或用户组。常见的 pods, services, deployments 等都是属于某一个 namespace 的（默认是default），而 Node, PersistentVolumes 等则不属于任何 namespace。



## Deployment

Deployment（部署）确保任意时间都有指定数量的 Pod 副本在运行。如果为某个 Pod 创建了Deployment 并且指定3个副本，它会创建3个 Pod，并且持续监控它们。如果某个 Pod 不响应，那么 Deployment 会替换它，保持总数为3.

如果之前不响应的 Pod 恢复了，现在就有4个 Pod 了，那么 Deployment 会将其中一个终止保持总数为3。如果在运行中将副本总数改为5，Deployment 会立刻启动2个新 Pod，保证总数为5。Deployment 还支持回滚和滚动升级。

当创建 Deployment 时，需要指定两个东西：

+ Pod模板：用来创建 Pod 副本的模板
+ Label标签：Deployment 需要监控的 Pod 的标签。



## Service

![img](https://d33wubrfki0l68.cloudfront.net/cc38b0f3c0fd94e66495e3a4198f2096cdecd3d5/ace10/docs/tutorials/kubernetes-basics/public/images/module_04_services.svg)

Service（服务）是应用服务的抽象，通过 labels 为应用提供负载均衡和服务发现。匹配 labels 的Pod IP 和端口列表组成 endpoints，由 kube-proxy 负责将服务 IP 负载均衡到这些endpoints 上。

每个 Service 都会自动分配一个 cluster IP（仅在集群内部可访问的虚拟地址）和 DNS 名，其他容器可以通过该地址或 DNS 来访问服务，而不需要了解后端容器的运行。





# API

Kubernetes 多组件之间的通信原理：

+ apiserver 负责 etcd 存储的所有操作，且只有 apiserver 才直接操作 etcd
+ apiserver 对内（集群中的其他组件）和对外（用户）提供统一的 REST API，其他组件均通过 apiserver 进行通信
  + controller manager、scheduler、kube-proxy 和 kubelet 等均通过 apiserver watch API 监测资源变化情况，并对资源作相应的操作
  + 所有需要更新资源状态的操作均通过 apiserver 的 REST API 进行

比如最典型的创建 Pod 的流程：![k8s pod](https://www.qikqiak.com/k8s-book/docs/images/k8s-pod-process.png)

+ 用户通过 REST API 创建一个 Pod
+ apiserver 将其写入 etcd
+ scheduluer 检测到未绑定 Node 的 Pod，开始调度并更新 Pod 的 Node 绑定
+ kubelet 检测到有新的 Pod 调度过来，通过 container runtime 运行该 Pod
+ kubelet 通过 container runtime 取得 Pod 状态，并更新到 apiserver 中





# 对象管理

在 Kubernetes 中，所有对象都使用 manifest（yaml或json）来定义，比如一个简单的 nginx 服务可以定义为 nginx.yaml，它包含一个镜像为 nginx 的容器：

```yaml
apiVersion: v1
kind: Pod
metadata:  
  name: nginx  
  labels:    
    app: nginx
spec:  
  containers:  
  - name: nginx    
    image: nginx    
    ports:    
    - containerPort: 80
```



描述对象

```shell
# 列出资源
$ kubectl get
$ kubectl get pods

# 显示有关资源的详细信息
$ kubectl describe
$ kubectl describe pods

# 打印 pod 和其中容器的日志
$ kubectl logs
$ kubectl logs $POD_NAME

# 在 pod 中的容器上执行命令
$ kubectl exec
$ kubectl exec $POD_NAME env


```

