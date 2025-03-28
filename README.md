
## We can decribe our provided functions here ##

在这里了解markdown： https://markdown.com.cn/basic-syntax/line-breaks.html 
### hand control ###
control_hand( coordination : dict, label) 目标物体的坐标，物品类型 
    ..... 
    identify grasp type，识别抓握类型 
    get the 3d coordination of the object, 得到物品3d坐标 
    move hand to the object, 让手接近物品 
    hold the object, 抓住物品 
    ..... 
    return nothing 

hand_release ( coordination = (-1,-1,-1) ) 放手，如果没有输入坐标则原地放手 
    return nothing 
  
需要改进，增加函数返回值等请加载这里：  


      

