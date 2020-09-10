[playground](https://katacoda.com/courses/kubernetes/playground)提供了包含了两个节点（Master 和一个 Node）的k8s集群，可用于测试。

```shell

# describe 显示有关资源的详细信息
$ kubectl describe pods

# logs 打印 pod 和其中容器的日志
$ kubectl logs $POD_NAME

# exec 在 pod 中的容器上执行命令
$ kubectl exec $POD_NAME env

```

## 演示

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
# deployment.apps/hello-minikube created

# 3. 将 deployment 作为 service 公开
$ kubectl expose deployment hello-minikube --type=NodePort --port=8080
# service/hello-minikube exposed

# 4. 查看 pod 状态
$ kubectl get pod
# NAME                              READY   STATUS    RESTARTS   AGE
# hello-minikube-5d9b964bfb-x7f5j   1/1     Running   0          93s

# 5. 获取 service 的 url
$ minikube service hello-minikube --url
# http://192.168.99.100:32481
```

```
Hostname: hello-minikube-5d9b964bfb-x7f5j

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
# 6. 删除 service
$ kubectl delete services hello-minikube
# service "hello-minikube" deleted

# 7. 删除 deployment
$ kubectl delete deployment hello-minikube
# deployment.extensions "hello-minikube" deleted

# 8. 停止本地 Minikube 集群
$ minikube stop
# ✋  Stopping node "minikube"  ...
# 🛑  1 nodes stopped.

```