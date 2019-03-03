### 发型模型

1.发型预测的cpp文件需要添加：

```c++
const float mean_vals[3] = {104.f, 117.f, 123.f};
const float norm_vals[3] = {0.017f, 0.017f, 0.017f};
in.substract_mean_normalize(mean_vals, norm_vals);
```

### 年龄模型

0.年龄预测的cpp文件：

```c++
const float mean_vals[3] = {104.f, 117.f, 123.f};
in.substract_mean_normalize(mean_vals, 0);
ncnn::Extractor ex = age.create_extractor();
ex.input("data", in);
ncnn::Mat out;
ex.extract("pred", out);

```
1.修改src文件夹下CMakeLists.txt

第186行添加：

ncnn_add_layer(NeuralDecisionRegForest)

2.在src/layer目录下添加文件

src/neuraldecisionregforest.cpp

src/neuraldecisionregforest.h

3.修改 tools/caffe/caffe2ncnn.cpp

详情看：caffe2ncnn.cpp

4.修改tools/caffe/caffe.proto

311行添加：

optional NeuralDecisionForestParameter neural_decision_forest_param = 1149;
optional LDLMetricParameter ldl_metric_param = 1150;
optional CSParameter cs_param = 1151;

419行添加：

message NeuralDecisionForestParameter {
  optional uint32 depth = 1 [default = 3];
  optional uint32 num_trees = 2 [default = 1];
  optional uint32 num_classes = 3 [default = 2];
  optional uint32 iter_times_class_label_distr = 4 [default = 20];
  optional uint32 iter_times_in_epoch = 7 [default = 20];
  optional string record_filename = 5 [default="Forest.Record"];
  optional uint32 axis = 6 [default = 1];
  optional bool debug_gpu = 8 [default = false];
  optional bool use_gpu = 9 [default = true];
  optional uint32 all_data_vec_length = 10 [default = 5];
  optional bool drop_out = 11 [default = false];
  optional string init_filename = 12 [default = "Leafnode.Init"];
  optional float scale = 13 [default = 100.0];
}

message LDLMetricParameter {
  enum LDLMetricType {
​    KLD = 1;
​    Clark = 2;
​    Chebyshev = 3;
​    Canberra = 4;
​    Cosine = 5;
​    Inter = 6;
​    Fidelity = 7;
​    Euclid = 8;
​    Soren = 9;
​    Square = 10;
  }
  optional LDLMetricType metric_type = 1 [default = KLD];
}

message CSParameter {
  optional int32 lll = 1 [default = 5];
}