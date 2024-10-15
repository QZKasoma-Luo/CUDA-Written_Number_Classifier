#ifndef HIDDEN_LAYER_H
#define HIDDEN_LAYER_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct HiddenLayer HiddenLayer;

    // 创建隐藏层
    HiddenLayer *createHiddenLayer(int inputSize, int hiddenSize);

    // 初始化权重和偏置
    void initializeLayer(HiddenLayer *layer);

    // 前向传播
    void forwardPass(HiddenLayer *layer, float *input, int batchSize);

    // 反向传播
    void backwardPass(HiddenLayer *layer, float *gradOutput, float learningRate, int batchSize);

    // 获取输出
    void getOutput(HiddenLayer *layer, float *output, int batchSize);

    // 释放资源
    void destroyHiddenLayer(HiddenLayer *layer);

    // 添加以下函数来访问结构体成员
    float *getWeights(HiddenLayer *layer);
    float *getBiases(HiddenLayer *layer);

#ifdef __cplusplus
}
#endif

#endif // HIDDEN_LAYER_H