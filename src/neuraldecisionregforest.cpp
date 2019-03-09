
// neuraldecisionregforest.cpp
#include "neuraldecisionregforest.h"
#include "layer_type.h"
#include <math.h>
#include <vector>
namespace ncnn{

DEFINE_LAYER_CREATOR(NeuralDecisionRegForest)

NeuralDecisionRegForest::NeuralDecisionRegForest()
{
	one_blob_only = true;
    support_inplace = true;
    support_vulkan = false;
}

// new routine for loading parameters
int NeuralDecisionRegForest::load_param(const ParamDict& pd)
{
    depth = pd.get(0, 6);
    num_trees = pd.get(1, 5); 
    num_classes = pd.get(2, 1); 
    axis = pd.get(6,1);
    scale_ = pd.get(13,100.f);
    mean_size = pd.get(14,1);
    sigma_size = pd.get(15,1);
    sub_dim_size = pd.get(16,1);

    num_leaf_nodes_per_tree_ = (int)pow(2, depth - 1);
    num_split_nodes_per_tree_ = num_leaf_nodes_per_tree_ - 1;
    num_nodes_pre_tree_ = num_leaf_nodes_per_tree_ + num_split_nodes_per_tree_;
    return 0;// return zero if success
}

int NeuralDecisionRegForest::load_model(const ModelBin& mb)
{
    mean_data = mb.load(mean_size, 1);
    sigma_data = mb.load(sigma_size, 1);
    sub_dim_data = mb.load(sub_dim_size, 1);
    if (mean_data.empty())
        return -100;
    if (sigma_data.empty())
        return -100;
    if (sub_dim_data.empty())
        return -100;   
    return 0;
}
int NeuralDecisionRegForest::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
	//bottom_top_blob dims:1, w:128, h:1, channels:1
	Layer* sigmoid = ncnn::create_layer(ncnn::LayerType::Sigmoid);
	sigmoid->forward_inplace(bottom_top_blob,opt);  
	delete sigmoid;          

	int off,off2;
    ncnn::Mat sub_dimensions_=sub_dim_data.reshape(num_trees,num_split_nodes_per_tree_);
    Mat routing_split_prob_=Mat(num_trees, num_split_nodes_per_tree_);
    Mat routing_leaf_prob_=Mat(num_trees, num_leaf_nodes_per_tree_);
    Mat forest_prediction_data=Mat(num_classes, 1);

	for (int t = 0; t < num_trees; ++t)
	{
		for (int j = 0; j < num_split_nodes_per_tree_; ++j)
		{
			if(j==0)
				routing_split_prob_.row(t)[j]=1.0;
			else 
				routing_split_prob_.row(t)[j]=0.0;
		}
	}

	for (int t = 0; t < num_trees; ++t)
	{
		for (int j = 0; j < num_split_nodes_per_tree_; ++j)
		{
			int current_offset = j;
			int dim_offset = (int)sub_dimensions_.row(t)[j];
			int left_child_offset = 2 * current_offset + 1;
			int right_child_offset = 2 * current_offset + 2;
			if (right_child_offset < num_split_nodes_per_tree_)
			{
				off=(t)*num_split_nodes_per_tree_+ left_child_offset;
				off2=(t)*num_split_nodes_per_tree_+ right_child_offset;
				routing_split_prob_[off] = routing_split_prob_.row(t)[current_offset] * bottom_top_blob[dim_offset];
				routing_split_prob_[off2] = routing_split_prob_.row(t)[current_offset] * ( 1.0 - bottom_top_blob[dim_offset]);
			}
			else
			{
				left_child_offset -= num_split_nodes_per_tree_;
				right_child_offset -= num_split_nodes_per_tree_;
				off=(t)*num_leaf_nodes_per_tree_+ left_child_offset;
				off2=(t)*num_leaf_nodes_per_tree_+ right_child_offset;
				routing_leaf_prob_[off] = routing_split_prob_.row(t)[current_offset] * bottom_top_blob[dim_offset];
				routing_leaf_prob_[off2] = routing_split_prob_.row(t)[current_offset] * ( 1.0 - bottom_top_blob[dim_offset]);
			}
		}
	}
	{
		routing_leaf_prob_=routing_leaf_prob_.reshape(num_trees*num_leaf_nodes_per_tree_);
		int len=mean_data.w;
		len=num_trees*num_leaf_nodes_per_tree_;
		for(int c=0;c<num_classes;c++)
			for(int i=0;i<len;i++)
				forest_prediction_data[c] += routing_leaf_prob_[i] * mean_data[i];

		for (int c = 0; c < num_classes; c++)
			{
				bottom_top_blob[c]= forest_prediction_data[c] *scale_ / num_trees;
			}
	}		

	return 0;
}
}
