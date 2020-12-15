> 参考[Kubernetes 初体验](https://www.qikqiak.com/k8s-book/docs/14.Kubernetes%E5%88%9D%E4%BD%93%E9%AA%8C.html)，[学习Kubernetes基础知识](https://kubernetes.io/zh/docs/tutorials/kubernetes-basics/)

[toc]

# 什么是Kubernetes?

> 参考[Kubernetes 是什么？](https://kubernetes.io/zh/docs/concepts/overview/what-is-kubernetes/)

Kubernetes 是一个可移植的、可扩展的开源平台，用于管理容器化的工作负载和服务，可促进声明式配置和自动化。 Kubernetes 拥有一个庞大且快速增长的生态系统。Kubernetes 的服务、支持和工具广泛可用。

名称 **Kubernetes** 源于希腊语，意为“舵手”或“飞行员”。Google 在 2014 年开源了 Kubernetes 项目。 Kubernetes 建立在 [Google 在大规模运行生产工作负载方面拥有十几年的经验](https://research.google/pubs/pub43438) 的基础上，结合了社区中最好的想法和实践。





# 架构

![](https://d33wubrfki0l68.cloudfront.net/7016517375d10c702489167e704dcb99e570df85/7bb53/images/docs/components-of-kubernetes.png)



## 集群

一个 Kubernetes 集群由一组被称作节点的机器组成。这些节点上运行 Kubernetes 所管理的容器化应用。集群具有至少一个工作节点和至少一个主节点。

![](https://d33wubrfki0l68.cloudfront.net/2475489eaf20163ec0f54ddc1d92aa8d4c87c96b/e7c81/images/docs/components-of-kubernetes.svg)



## Master

![k8s cluster](https://www.qikqiak.com/k8s-book/docs/images/k8s-cluster.png)

Master（主节点）是 Kubernetes 集群的控制节点，也称为 Control Plane（控制面），<u>负责整个集群的管理和控制</u>。基本上所有的控制命令都是发送给它，然后由它来负责具体的执行过程。Master 节点通常占据一个独立的x86服务器，一个主要的原因是它太重要了，如果它宕机或者不可用，那么我们所有的控制命令都将失效。

Master 节点上运行着以下一组关键进程：

+ Kubernetes API Server (kube-apiserver)：所有资源的增删改查的入口，集群控制的入口，提供 HTTP REST 服务，与 Node 以及 kubeclt 通信
+ Kubernetes Controller Manager (kube-controller-manager)：集群中所有资源对象的自动化控制中心
  + Node Controller（节点控制器）: 负责在节点出现故障时进行通知和响应。
  + Replication Controller（副本控制器）: 负责为系统中的每个副本控制器对象维护正确数量的 Pod。
  + Endpoints Controller（端点控制器）: 填充端点(Endpoints)对象(即加入 Service 与 Pod)。
  + Service Account & Token Controllers（服务帐户和令牌控制器）: 为新的命名空间创建默认帐户和 API 访问令牌.
+ Kubernetes Scheduler (kube-scheduler)：负责 Pod 的调度，即监视新创建的 Pod 并为其分配 Node
+ etcd：保存 Kubernetes 所有集群数据的后台数据库
+ cloud-controller-manager：云控制器管理器是 1.8 的 alpha 特性。在未来发布的版本中，这是将 Kubernetes 与任何其他云集成的最佳方式。



## Node

![](https://d33wubrfki0l68.cloudfront.net/5cb72d407cbe2755e581b6de757e0d81760d5b86/a9df9/docs/tutorials/kubernetes-basics/public/images/module_03_nodes.svg)

Node（工作节点）是 Kubernetes 中的参与计算的机器（除了 Master 之外的所有机器），可以是虚拟机或物理计算机。每个 Node 由 Master 管理，都会被 Master 分配一些工作负载（Docker容器）。Master 的自动调度考量了每个 Node 上的可用资源，若某个 Node 宕机，其上的工作负载会被 Master 自动转移到其他 Node 上去。

每个 Node 节点上运行着以下一组关键进程：

+ kubelet，负责与 Master 的通信，并且管理 Pod 的创建、启动、监控、重启、销毁等工作
+ container runtime（如 Docker），负责运行容器的软件
+ kube-proxy，实现 Service 的服务发现和负载均衡


Node 节点可以在运行期间动态增加到 Kubernetes 集群中，条件是在这个 Node 上配置和启动了上述关键进程。在默认情况下 kubelet 会向 Master 注册自己，这也是 Kubernetes 推荐的 Node 管理方式。一旦 Node 被纳入集群管理范围，kubelet 就会定时向 Master 汇报自身的情报，例如操作系统、Docker版本、机器的CPU和内存使用，哪些 Pod 在运行等，这样 Master 就可以获知每个 Node 的资源使用情况，并实现高效均衡的资源调度策略。如果某个 Node 超过指定时间不上报信息，那么会被 Master 判定为失去连接，Master 将 Node 的状态标记为不可用(Not Ready)，并触发“工作负载大转移”的自动流程。



## Pod

![](https://d33wubrfki0l68.cloudfront.net/fe03f68d8ede9815184852ca2a4fd30325e5d15a/98064/docs/tutorials/kubernetes-basics/public/images/module_03_pods.svg)

Pod 是一组紧密关联的容器集合，它们共享 PID、IPC、存储、网络 和 UTS namespace，是Kubernetes 调度的基本单元。Pod 的设计理念是支持多个容器在一个 Pod 中共享网络和文件系统，可以通过<u>进程间通信和文件共享</u>这种简单高效的方式组合完成服务。Pod 中的容器始终位于同一位置并且共同调度，并在同一工作节点上的共享上下文中运行。

Node 中的每个 Pod 都分配了唯一的IP地址，称为Pod IP。Kubernetes 要求底层网络支持集群内任意两个 Pod 之间的TCP/IP直接通信，这通常使用虚拟二层网络技术来实现，因此我们需要记住，一个 Pod 里的容器与其它 Node 的 Pod 的容器可以直接通信。

Pod 一旦被创建，就会被放入到 etcd 中存储，并由 Master 调度到某个具体的 Node 上进行绑定(binding)，随后该 Pod 由 对应 Node 上的 kubelet 进程实例化为一组相关的 Docker 容器并启动。如果 Pod 中的某个容器停止，Kubernetes 会自动检测到这个问题并且重新启动这个 Pod；如果 Pod 所在的 Node 宕机，则会将该 Node 上的所有 Pod 重新调度到其它 Node 上。

下面是一个 Pod 的配置文件：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-demo
spec:
  containers:
  - name: nginx
    image: nginx:1.14.2
    ports:
    - containerPort: 80  # 暴露容器的80端口
```



## Label

Label（标签）是识别 Kubernetes 对象的标签，以 key/value 的方式附加到对象上（key最长不能超过63字节，value 可以为空，也可以是不超过253字节的字符串）。 Label 可以附加到各种资源对象上，例如 Node、Pod、Service、RC等。Label 不提供唯一性，并且实际上经常是很多对象（如Pods）都使用相同的 label 来标志具体的应用。

我们可以通过给指定的资源对象捆绑一个或多个不同的 Label 来实现多维度的资源分组管理功能，以便于灵活、方便地进行资源分配、调度、配置、部署等管理工作。一些常用的 Label 示例如下：

+ 版本标签：`release: stable`
+ 环境标签：`environment: dev`
+ 架构标签：`tier: frontend`
+ 分区标签：`partition: clientA`
+ 质量管控标签：`track: daily`

Label 定义好后其他对象可以使用 Label Selector 来查询和筛选拥有某些 Label 的资源对象（比如Service 用 Label 来选择一组 Pod）。Label Selector支持以下几种方式：

+ 等式，如`app=nginx`匹配所有具有标签`app=nginx`的资源对象，`env!=production`匹配所有<u>不具有标签`env=production`的资源对象</u>
+ 集合，如`app in (nginx, mysql)`匹配所有具有标签`app=nginx`或`app=mysql`的资源对象，`env not in (production, qa)`匹配所有不具有标签`env=production`和`env=qa`的资源对象；`partition`匹配所有具有键为`partition`的标签的资源对象，`!partition`匹配所有不具有键为`partition`的标签的资源对象
+ 多个label（它们之间是AND关系），如`app=nginx,env=test`

Label Selector 在 Kubernetes 中的重要使用场景有以下几处：

+ kube-controller 进程通过资源对象 RC 上定义的 Label Selector 来筛选要监控的 Pod 副本的数量，从而控制副本数量始终符合预期设定
+ kube-proxy 进程通过 Service 的 Label Selector 来选择对应的 Pod，自动建立每个 Service 到对应 Pod 的请求转发路由表，从而实现 Service 的负载均衡机制

以下命令查找有指定标签的 Pod 对象

```shell
$ kubectl get pods -l environment=production,tier=frontend
# 或
$ kubectl get pods -l 'environment in (production),tier in (frontend)'
```



## Replication Controller (deprecated)

RC 是 Kubernetes 系统的核心概念之一，它的定义包含以下几个部分：

+ Pod 期待的副本数(replicas)
+ 用于筛选目标 Pod 的 Label Selector
+ 当 Pod 少于预期数量时，用于创建新 Pod 的 Pod 模板

下面是一个完整的 RC 定义的例子，它确保拥有`app: nginx`标签的这个 Pod 在整个 Kubernetes 集群中始终只有3个副本。

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: nginx
spec:
  replicas: 3   # 3个副本
  selector:
    app: nginx  # 对于标签app: nginx
  template:
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



当我们定义了一个 RC 并提交到 Kubernetes 集群中后， Master 节点上的 Controller Manager 组件就得到通知，定期巡检系统中当前运行的目标 Pod，如果运行的副本多于期望值则停掉一些 Pod，少于则自动创建一些 Pod。



## Deployment

Deployment（部署）确保任意时间都有指定数量的 Pod 副本在运行。与 RC 类似，Master 节点上的 Controller Manager 组件会定期检查系统中当前运行的目标 Pod，如果运行的副本多于期望值则停掉一些 Pod，少于则自动创建一些 Pod。

以下是 Deployment 的典型使用场景：

+ [创建 Deployment 以将 ReplicaSet 上线](https://kubernetes.io/zh/docs/concepts/workloads/controllers/deployment/#creating-a-deployment)。 ReplicaSet 在后台创建 Pods。 检查 ReplicaSet 的上线状态，查看其是否成功。
+ 通过更新 Deployment 的 PodTemplateSpec，[声明 Pod 的新状态](https://kubernetes.io/zh/docs/concepts/workloads/controllers/deployment/#updating-a-deployment) 。 新的 ReplicaSet 会被创建，Deployment 以受控速率将 Pod 从旧 ReplicaSet 迁移到新 ReplicaSet。 每个新的 ReplicaSet 都会更新 Deployment 的修订版本。
+ 如果 Deployment 的当前状态不稳定，[回滚到较早的 Deployment 版本](https://kubernetes.io/zh/docs/concepts/workloads/controllers/deployment/#rolling-back-a-deployment)。 每次回滚都会更新 Deployment 的修订版本。
+ [扩大 Deployment 规模以承担更多负载](https://kubernetes.io/zh/docs/concepts/workloads/controllers/deployment/#scaling-a-deployment)。
+ [暂停 Deployment](https://kubernetes.io/zh/docs/concepts/workloads/controllers/deployment/#pausing-and-resuming-a-deployment) 以应用对 PodTemplateSpec 所作的多项修改， 然后恢复其执行以启动新的上线版本。
+ [使用 Deployment 状态](https://kubernetes.io/zh/docs/concepts/workloads/controllers/deployment/#deployment-status) 来判定上线过程是否出现停滞。
+ [清理较旧的不再需要的 ReplicaSet](https://kubernetes.io/zh/docs/concepts/workloads/controllers/deployment/#clean-up-policy) 。

下面是一个 Deployment 示例，其创建了一个 ReplicaSet，负责启动三个 `nginx` Pod：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3       # 3个Pod副本
  selector:         # 查找方式
    matchLabels:
      app: nginx
  template:         # 生成模板
    metadata:
      labels:
        app: nginx  # 标签
    spec:
      containers:   # 容器配置
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
```



## Service

![img](https://d33wubrfki0l68.cloudfront.net/cc38b0f3c0fd94e66495e3a4198f2096cdecd3d5/ace10/docs/tutorials/kubernetes-basics/public/images/module_04_services.svg)

> Pod 是有生命周期的，如果使用 Deployment 来运行应用程序，那么它可以动态创建和销毁 Pod ，而每个 Pod 都有自己的 IP 地址。这时的一个问题是， 如果一组 Pod（称为“后端”）为集群内的其他 Pod（称为“前端”）提供功能， 那么前端如何找出并跟踪要连接的 IP 地址以使用后端的功能？

Service 提供了一种将运行在一组 Pod 上的应用程序公开为网络服务（通常称为微服务）的抽象方法，通过 Label 为应用提供负载均衡和服务发现。匹配 Label 的 Pod IP 和端口列表组成 endpoints，由 kube-proxy 负责将服务 IP 负载均衡到这些 endpoints 上。

> 一个 Service 下有一组 Pod，其中每个 Pod 提供一个 endpoint，即 Service 转发请求的目的地。

每个 Service 都会自动分配一个 cluster IP（仅在集群内部可访问的虚拟地址）和 DNS 名，其他容器可以通过该地址或 DNS 来访问服务，而不需要了解后端容器的运行。

Service 有以下`type`的方式暴露

+ *ClusterIP* (默认) - 在集群的内部 IP 上公开 Service 。这种类型使得 Service 只能从集群内访问。
+ *NodePort* - 使用 NAT 在集群中每个选定 Node 的相同端口上公开 Service ，可以使用`<NodeIP>:<NodePort>` 从集群外部访问 Service。是 ClusterIP 的超集。
+ *LoadBalancer* - 在当前云中创建一个外部负载均衡器(如果支持的话)，并为 Service 分配一个固定的外部IP。是 NodePort 的超集。
+ *ExternalName* - 通过返回带有该名称的 CNAME 记录，使用任意名称(由 spec 中的`externalName`指定)公开 Service。不使用代理。这种类型需要`kube-dns`的v1.7或更高版本。



假定有一组 Pod，它们对外暴露了 9376 端口，同时还被打上 `app=MyApp` 标签，以下配置创建一个 Service 对象

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: MyApp
  ports:
    - protocol: TCP
      port: 80
      targetPort: 9376
```

该 Service 对象会将请求代理到使用 TCP 端口 9376，并且具有标签 `app=MyApp` 的 Pod 上。 



## Namespace

namespace 通常用于实现多租户的资源隔离：通过将集群内部的资源对象贴上不同的 namespace 的标签，形成逻辑上分组的不同项目、小组或用户组，便于不同的分组在共享使用整个集群的资源的同时还能被分别管理。

Kubernetes 集群在启动后，会创建一个名为`default`的 namespace。接下来如果不特别指明 namespace，则创建的所有 Pod, Deployment, Service 都将被创建到默认 namespace 中。

namespace 的定义很简单，下面的 yaml 定义了名为 development 的 namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: development
```

一旦创建了 namespace，我们在创建资源对象时就可以指定这个资源对象属于哪个 namespace，下面的 yaml 将定义的 Pod 放入 development 这个 namespace 中

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
  namespace: development
spec:
  containers:
  - name: nginx
    image: nginx:1.7.9
    ports:
    - containerPort: 80
```

要查看这个 Pod，则需要加入`--namespace`参数

```shell
$ kubectl get pods --namespace=development
```

我们给每个租户创建一个 namespace 来实现多租户的资源隔离时，还能结合 Kubernetes 的资源限额管理，限定不同租户能占用的资源，例如 CPU、内存使用量等。



## Annotation

Annotation 与 Label 类似，也使用 key/value 键值对的形式定义。但不同的是 Annotation 是用户任意定义的附加信息，以便于外部工具查找。很多时候，Kubernetes 的模块会通过 Annotation 的方式标记资源对象的一些特殊信息。





# 对象管理

> 参考：https://kubernetes.io/zh/docs/concepts/overview/working-with-objects/

## Kubernetes 对象

在 Kubernetes 系统中，Kubernetes 对象是持久化的实体， Kubernetes 使用这些实体去表示整个集群的状态。特别地，它们描述了如下信息：

+ 哪些容器化应用在运行（以及在哪些节点上）
+ 可以被应用使用的资源
+ 关于应用运行时表现的策略，比如重启策略、升级策略，以及容错策略

Kubernetes 对象是 “目标性记录” —— 一旦创建对象，Kubernetes 系统将持续工作以确保对象存在。 创建对象本质上是在告知 Kubernetes 系统，所需要的集群工作负载看起来是什么样子的， 这就是 Kubernetes 集群的 **期望状态（Desired State）**。

操作 Kubernetes 对象 —— 无论是创建、修改，或者删除 —— 需要使用 Kubernetes API。 比如当使用 `kubectl` 命令行接口时，CLI 会执行必要的 Kubernetes API 调用。



## 定义对象

创建 Kubernetes 对象时，必须提供对象的`spec`，用来描述该对象的期望状态， 以及关于对象的一些基本信息（例如名称）。当使用 Kubernetes API 创建对象时（或者直接创建，或者基于`kubectl`）， API 请求必须在请求体中包含 JSON 格式的信息。 **大多数情况下，需要在 .yaml 文件中为 `kubectl` 提供这些信息**。 `kubectl` 在发起 API 请求时，将这些信息转换成 JSON 格式。

这里有一个 `.yaml` 示例文件，展示了 Kubernetes Deployment 的必需字段和对象规约：

```yaml
apiVersion: apps/v1   # for versions before 1.9.0 use apps/v1beta2
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  selector:
    matchLabels:
      app: nginx
  replicas: 2   # tells deployment to run 2 pods matching the template
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

使用类似于上面的 `.yaml` 文件来创建 Deployment的一种方式是使用 `kubectl` 命令行接口中的 `kubectl apply` 命令， 将 `.yaml` 文件作为参数。下面是一个示例：

```shell
kubectl apply -f https://k8s.io/examples/application/deployment.yaml --record
```



## 管理对象

> kubectl的所有命令参见https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands



### Imperative commands(命令式命令)

命令式命令直接对集群中的活动对象进行操作，这是在集群中运行一次性任务的最简单方法。例如：

```shell
# 创建 Deployment 对象来运行 nginx 容器的实例
$ kubectl run nginx --image nginx
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

namespace 适用于存在多个用户的场景，提供了在多个用户之间划分集群资源的一种方法。在 Kubernetes 未来版本中，相同 namespace 中的对象默认将具有相同的访问控制策略。

namespace 为名称提供了一个范围。资源的名称需要在 namespace 内是唯一的。  namespace 不能相互嵌套，每个 Kubernetes 资源只能在一个 namespace 中。





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

