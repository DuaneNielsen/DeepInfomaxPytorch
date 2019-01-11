# Deep InfoMax Pytorch

Pytorch implementation of Deep InfoMax
https://arxiv.org/abs/1808.06670

Encoding data by maximimizing mutual information between the latent space and in this case, CIFAR 10 images.

Ported most of the code from rcallands chainer implementation.  Thanks buddy!  https://github.com/rcalland/deep-INFOMAX

Pytorch implementation by the research team [here](https://github.com/rdevon/DIM)

### Current Results (work in progress)


|              |airplane |automobile | bird | cat |    deer|   dog |    frog|   horse|  ship|   truck|
|-----------------|-------|--------|-------|-------|-------|-------|-------|-------|-------|------|
|Fully supervised |0.7780 | 0.8907 | 0.6233| 0.5606| 0.6891| 0.6420| 0.7967| 0.8206| 0.8619| 0.8291
|DeepInfoMax-Local|0.6120 | 0.6969 | 0.4020| 0.4226| 0.4917| 0.5806| 0.6871| 0.5806| 0.6855| 0.5647
                   

![alt_text](images/Figure_1.png "Figure 1")

**Figure 1**  
**Top:** a red lamborghini, **Middle:** 10 closest images in the latent space (L2 distance), **Bottom:** 10 farthest images in the latent space.

Some more results..

![alt_text](images/Figure_2.png "Result")

![alt_text](images/Figure_3.png "Result")

![alt_text](images/Figure_4.png "Result")

![alt_text](images/Figure_5.png "Result")



