//添加在第595行

else if(layer.type()=="NeuralDecisionRegForest")
{
    const caffe::LayerParameter& binlayer = net.layer(netidx);

    const caffe::BlobProto& mean_blob = binlayer.blobs(0);
    const caffe::BlobProto& sigma_blob = binlayer.blobs(1);
    const caffe::BlobProto& sub_dim_blob = binlayer.blobs(2);
    fwrite(mean_blob.data().data(), sizeof(float), mean_blob.data_size(), bp);
    fwrite(sigma_blob.data().data(), sizeof(float), sigma_blob.data_size(), bp);
    fwrite(sub_dim_blob.data().data(), sizeof(float), sub_dim_blob.data_size(), bp);

    const caffe::NeuralDecisionForestParameter& mdrf_param = layer.neural_decision_forest_param();
    int depth = mdrf_param.depth();
    int num_trees = mdrf_param.num_trees();
    int num_classes = mdrf_param.num_classes();
    fprintf(pp, " 0=%d", mdrf_param.depth());
    fprintf(pp, " 1=%d", mdrf_param.num_trees());
    fprintf(pp, " 2=%d", mdrf_param.num_classes());
    fprintf(pp, " 14=%d", (int)mean_blob.data_size());
    fprintf(pp, " 15=%d", (int)sigma_blob.data_size());
    fprintf(pp, " 16=%d", (int)sub_dim_blob.data_size());
}