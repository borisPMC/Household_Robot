
## We can decribe our provided functions here ##
<br>

### hand control ###
control_hand( coordination : dict, label) 目标物体的坐标，物品类型 <br>
    ..... <br>
    identify grasp type，识别抓握类型 <br>
    get the 3d coordination of the object, 得到物品3d坐标 <br>
    move hand to the object, 让手接近物品 <br>
    hold the object, 抓住物品 <br>
    ..... <br>
    return nothing <br>

hand_release ( coordination = (-1,-1,-1) ) 放手，如果没有输入坐标则原地放手 <br>
    return nothing <br>
  <br>
需要改进，增加函数返回值等请加载这里：  <br>


Message to Boris:

After installation of urx, please comment out urx/robot.py Line 204-205

