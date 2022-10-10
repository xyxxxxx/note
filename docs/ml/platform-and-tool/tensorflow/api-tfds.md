# tfds

`tfds` 模块定义了一系列 TensorFlow 可以直接使用的数据集的集合。每个数据集都定义为一个 `tfds.core.DatasetBuilder` 实例，该实例封装了下载数据集、构建输入流水线的逻辑，也包含了数据集的文档。

## as_dataframe()

将数据集转换为 Pandas dataframe。

```python
tfds.as_dataframe(
    ds: tf.data.Dataset,
    ds_info: Optional[tfds.core.DatasetInfo] = None
) -> StyledDataFrame
# ds        要转换为Pandas dataframe的`Dataset`实例,其中样本不应分批
# ds_info   `DatasetInfo`实例,用于帮助改善格式
```

## as_numpy()

将数据集转换为 NumPy 数组。

```python
tfds.as_numpy(
    dataset: Tree[TensorflowElem]
) -> Tree[NumpyElem]
```

```python
ds = tfds.load(name="mnist", split="train")
ds_numpy = tfds.as_numpy(ds)  # Convert `tf.data.Dataset` to Python generator
for ex in ds_numpy:
  # `{'image': np.array(shape=(28, 28, 1)), 'labels': np.array(shape=())}`
  print(ex)
```

## build()

通过数据集名称获取一个 `DatasetBuilder` 实例。

```python
tfds.builder(name: str, *, try_gcs: bool = False, **builder_kwargs) -> tfds.core.DatasetBuilder
# name                数据集名称,需要是`DatasetBuilder`中注册的名称.可以是'dataset_name'或'dataset_name/config_name'
#                     (对于有`Builderconfig`的数据集)
# try_gcs
# **builder_kwargs    传递给`DatasetBuilder`的关键字参数字典
```

```python
```

## core.BuilderConfig

## core.DatasetBuilder

## core.DatasetInfo

## load()

加载已命名的数据集为一个 `tf.data.Dataset` 实例。

```python
tfds.load(
    name: str,
    *,
    split: Optional[Tree[splits_lib.Split]] = None,
    data_dir: Optional[str] = None,
    batch_size: tfds.typing.Dim = None,
    shuffle_files: bool = False,
    download: bool = True,
    as_supervised: bool = False,
    decoders: Optional[TreeDict[decode.Decoder]] = None,
    read_config: Optional[tfds.ReadConfig] = None,
    with_info: bool = False,
    builder_kwargs: Optional[Dict[str, Any]] = None,
    download_and_prepare_kwargs: Optional[Dict[str, Any]] = None,
    as_dataset_kwargs: Optional[Dict[str, Any]] = None,
    try_gcs: bool = False
)
# name                数据集名称,需要是`DatasetBuilder`中注册的名称.可以是'dataset_name'或
#                     'dataset_name/config_name'(对于有`Builderconfig`的数据集)
# split               加载的数据集部分,例如'train','test',['train','test'],'train[80%:]',etc.若为`None`,
#                     则返回部分名称到`Dataset`实例的字典
# data_dir            读/写数据的目录
# batch_size          批次规模,设定后会为样本增加一个批次维度
# shuffle_files       若为`True`,打乱输入文件
# download            若为`True`,在调用`DatasetBuilder.as_dataset()`之前调用
#                     `DatasetBuilder.download_and_prepare()`,如果数据已经在`data_dir`下,则不执行任何操作;
#                     若为`False`,则数据应当存在于`data_dir`下.
# as_supervised       若为`True`,则`Dataset`实例返回的每个样本是一个二元组`(input, label)`;若为`False`,
#                     则`Dataset`实例返回的每个样本是一个包含所有特征的字典,例如`{'input': ..., 'label': ...}`
# decoders
# read_config
# with_info           若为`True`,则返回元组`(Dataset, DatasetInfo)`,后者包含了数据集的信息
# builder_kwargs      传递给`DatasetBuilder`的关键字参数字典
# download_and_prepare_kwargs     传递给`DatasetBuilder.download_and_prepare()`的关键字参数字典
# as_dataset_kwargs               传递给`DatasetBuilder.as_dataset()`的关键字参数字典
# try_gcs
```

```python
>>> datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

```
