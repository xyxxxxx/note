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

Master（主节点）是 Kubernetes 集群的控制节点，也称为 Control Plane（控制面），负责整个集群的管理和控制，包含以下组件：

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

Deployment（部署）确保任意时间都有指定数量的 Pod 副本在运行。如果为某个 Pod 创建了Deployment 并且指定3个副本，它会创建3个 Pod，并且持续监控它们。如果某个 Pod 不响应，那么 Deployment 会替换它，保持总数为3。

如果之前不响应的 Pod 恢复了，现在就有4个 Pod 了，那么 Deployment 会将其中一个终止保持总数为3。如果在运行中将副本总数改为5，Deployment 会立刻启动2个新 Pod，保证总数为5。Deployment 还支持回滚和滚动升级。

当创建 Deployment 时，需要指定两个东西：

+ Pod模板：用来创建 Pod 副本的模板
+ Label标签：Deployment 需要监控的 Pod 的标签。



## Service

![img](https://d33wubrfki0l68.cloudfront.net/cc38b0f3c0fd94e66495e3a4198f2096cdecd3d5/ace10/docs/tutorials/kubernetes-basics/public/images/module_04_services.svg)

Service（服务）是应用服务的抽象，通过 labels 为应用提供负载均衡和服务发现。匹配 labels 的Pod IP 和端口列表组成 endpoints，由 kube-proxy 负责将服务 IP 负载均衡到这些endpoints 上。

每个 Service 都会自动分配一个 cluster IP（仅在集群内部可访问的虚拟地址）和 DNS 名，其他容器可以通过该地址或 DNS 来访问服务，而不需要了解后端容器的运行。





# API

Kubernetes控制面的核心是apiserver。apiserver 负责提供 HTTP API，以供用户、集群中的不同部分和集群外部组件相互通信。

Kubernetes API 使你可以查询和操纵 Kubernetes API 中对象（例如：Pod、Namespace、ConfigMap 和 Event）的状态。

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

## Kubernetes 对象

在 Kubernetes 系统中，Kubernetes 对象是持久化的实体， Kubernetes 使用这些实体去表示整个集群的状态。特别地，它们描述了如下信息：

+ 哪些容器化应用在运行（以及在哪些节点上）
+ 可以被应用使用的资源
+ 关于应用运行时表现的策略，比如重启策略、升级策略，以及容错策略

Kubernetes 对象是 “目标性记录” —— 一旦创建对象，Kubernetes 系统将持续工作以确保对象存在。 创建对象本质上是在告知 Kubernetes 系统，所需要的集群工作负载看起来是什么样子的， 这就是 Kubernetes 集群的 **期望状态（Desired State）**。

操作 Kubernetes 对象 —— 无论是创建、修改，或者删除 —— 需要使用 Kubernetes API。 比如当使用 `kubectl` 命令行接口时，CLI 会执行必要的 Kubernetes API 调用。



## 定义对象

创建 Kubernetes 对象时，必须提供对象的规约，用来描述该对象的期望状态， 以及关于对象的一些基本信息（例如名称）。当使用 Kubernetes API 创建对象时（或者直接创建，或者基于`kubectl`）， API 请求必须在请求体中包含 JSON 格式的信息。 **大多数情况下，需要在 .yaml 文件中为 `kubectl` 提供这些信息**。 `kubectl` 在发起 API 请求时，将这些信息转换成 JSON 格式。

这里有一个 `.yaml` 示例文件，展示了 Kubernetes Deployment 的必需字段和对象规约：

```yaml
apiVersion: apps/v1 # for versions before 1.9.0 use apps/v1beta2
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  selector:
    matchLabels:
      app: nginx
  replicas: 2 # tells deployment to run 2 pods matching the template
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
```

`.yaml` 文件的必需字段包括：

+ `apiVersion` - 创建该对象所使用的 Kubernetes API 的版本
+ `kind` - 想要创建的对象的类别
+ `metadata` - 帮助唯一性标识对象的一些数据，包括一个 `name` 字符串、UID 和可选的 `namespace`

你也需要提供对象的 `spec` 字段。 对象 `spec` 的精确格式对每个 Kubernetes 对象来说是不同的，包含了特定于该对象的嵌套字段。 [Kubernetes API 参考](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.19/) 能够帮助我们找到任何我们想创建的对象的 spec 格式。

使用类似于上面的 `.yaml` 文件来创建 Deployment的一种方式是使用 `kubectl` 命令行接口（CLI）中的 `kubectl apply` 命令， 将 `.yaml` 文件作为参数。下面是一个示例：

```shell
kubectl apply -f https://k8s.io/examples/application/deployment.yaml --record
```



## 管理对象

> kubectl的所有命令参见https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands

### Imperative commands(命令式命令)

命令式命令直接对集群中的活动对象进行操作，这是在集群中运行一次性任务的最简单方法。例如：

```shell
# 创建 Deployment 对象来运行 nginx 容器的实例
$ kubectl create deployment nginx --image=nginx
```

### Imperative object configuration(命令式对象配置)

命令式对象配置指定操作（创建，替换等），可选标志和至少一个文件名。指定的文件必须包含 YAML 或 JSON 格式的对象的完整定义。例如：

```shell
# 创建配置文件中定义的对象
$ kubectl create -f nginx.yaml

# 删除两个配置文件中定义的对象
$ kubectl delete -f nginx.yaml -f redis.yaml

# 通过覆盖活动配置来更新配置文件中定义的对象
$ kubectl replace -f nginx.yaml
```

### Declarative object configuration(声明式对象配置)

声明式对象配置对本地存储的对象配置文件进行操作，但是用户未定义要对该文件执行的操作。`kubectl` 会自动检测每个文件的创建、更新和删除操作。这使得配置可以在目录上工作，根据目录中配置文件对不同的对象执行不同的操作。例如：

```sh
# 处理 `configs` 目录中的所有对象配置文件，创建并更新活动对象
# 首先使用 `diff` 子命令查看将要进行的更改，然后进行应用
$ kubectl diff -f configs/
$ kubectl apply -f configs/
```



## 对象名称和ID

每个 Kubernetes 对象对象都有一个 name(名称) 来标识在同类资源中的唯一性，也有一个 UID 来标识在整个集群中的唯一性。



## namespace

Kubernetes 支持基于同一个物理集群的多个虚拟集群， 这些虚拟集群被称为 namespace。

### 何时使用多个namespace

namespace 适用于存在多个用户的场景，提供了在多个用户之间划分集群资源的一种方法。在 Kubernetes 未来版本中，相同名字空间中的对象默认将具有相同的访问控制策略。

名字空间为名称提供了一个范围。资源的名称需要在名字空间内是唯一的，但不能跨名字空间。 名字空间不能相互嵌套，每个 Kubernetes 资源只能在一个名字空间中。



## 标签和选择算符

Labels（标签）是附加到 Kubernetes 对象（比如 Pods）上的键值对，旨在用于指定对用户有意义且相关的对象的标识属性。标签可以在创建时附加到对象，随后可以随时添加和修改。 每个对象都可以定义一组键/值标签。

```json
"metadata": {
  "labels": {
    "key1" : "value1",
    "key2" : "value2"
  }
}
```