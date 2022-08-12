# 144_10decap
从VDI最大的十五个节点中挑选10个节点位置放置decap，结果为max VDI降低而total VDI并不一定降低，action选择有一定问题

loss function如下图

![ef1a5cca55ef1925a013fe99e240424](https://user-images.githubusercontent.com/89006608/184323010-89698521-d704-44a4-aaf0-8db4537e6f42.png)

初步判断是经验回放池的容量太小，并未有完全利用已有经验

每一个回合所得max/min reward和total reward以及每一个episode下每一个state的max/min VDI，这里的VDI是指所有节点VDI之和

![950cdf11e920409b560e474efc6b3c4](https://user-images.githubusercontent.com/89006608/184324110-8aa11258-63aa-4f6a-87a1-21a121ba17ca.png)

![4baf42ecdd616a4569fa8a2a83d25aa](https://user-images.githubusercontent.com/89006608/184330907-686ea1f3-3f1c-49d7-9c56-2ac22cb7ad5a.png)
![e386cb937bbd541688ce54ffa306b8f](https://user-images.githubusercontent.com/89006608/184324130-3e781b0d-c832-4909-a86d-096bda05d136.png)

从以上的几幅图可以看出，代码存在问题需要修改，，reward没有呈现逐渐增加的趋势，反而是loss function逐渐增大到不切实际的数值


