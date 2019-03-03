// mylayer.h

#ifndef LAYER_NEURALDECIDIONREGFOREST_H
#define LAYER_NEURALDECIDIONREGFOREST_H

#include "layer.h"
namespace ncnn{

class NeuralDecisionRegForest : public Layer
{
public:
    NeuralDecisionRegForest();
    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
public:
    int depth;
    int num_trees;
    int num_classes;
    int axis;
    float scale_;

    int mean_size;
    int sigma_size;
    int sub_dim_size;

    int num_leaf_nodes_per_tree_;
    int num_split_nodes_per_tree_;
    int num_nodes_pre_tree_;

    Mat mean_data;
    Mat sigma_data;
    Mat sub_dim_data;

};
}

#endif //NEURALDECIDIONREGFOREST