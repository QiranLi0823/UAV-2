# 文档说明

## 1. 环境配置
按照requirements.txt和require.txt(为防止安装依赖包出现不兼容问题，参考require.txt时从后往前安装，缺什么依赖包在两个txt中都能找到)

## 2. 文件说明

- 2.1 文件树如下
  -----workpath
    |-data                  # 用于生成angle必要数据
    |-mixformer             # 存放mixformer模型完整代码
    |-STTFormer(UAV)        # 存放STTFormer(UAV)模型完整代码
    |-pkl-A and pkl-B       # 存放mixformer与STTFormer训练结果
    |-torchligth            # 相关依赖包
    |-ensemble.py           # 融合模型训练的权重
    |-requirement.txt       # 环境所需要的依赖库
    |-require.txt           # 环境所需要的依赖库
    |-README.md             # 对本项目代码的说明
  

## 3. 数据位置
- 3.1 '文件说明'部分已说明数据集存放的位置
- 3.2 参数设置均在config文件中

## 4. 运行方式

- 4.1 在运行代码前，请先配置好'环境配置'中所需要的依赖的安装包
- 4.2 安装apex，获取地址(http://192.168.1.15:3000/QiranLi0823/apex.git),使用python setup.py install指令安装apex
- 4.3 相关数据模块的训练指令根据config文件中的运行指令
- 4.4 运行时要确定相关数据路径是否正确

注:安装apex时修改获取地址中的网关地址
   相关数据训练时注意在config文件中修改路径


## 5. 结果复现流程

- 5.1 确定环境配置以及相关依赖包以成功安装
- 5.2 将官方提供的数据集data放到与UAV-1同级目录中。
- 5.3 利用官方所给的代码处理出bone模态，祛除空数据的相关代码在data_process中
- 5.4 利用UAV-1文件中的data目录下的gen_angle_data.py运行文件生成angle模态（注意在代码中修改数据集路径）
- 5.5 将mixformer中的(_1,_2,_6,angle_1,motion_1,motion_2,motion_6)分别训练得到的训练日志，训练权重均保存在work_dir文件夹下
- 5.6 将STTFormer中的(angle,bone,joint,motion)分别训练得到的训练日志，训练权重均保存在work_dir文件夹下
- 5.7 对结果运行ensemble.py操作融合，融合后生成B数据集的置信度文件

```.
注:每个模型的config中都标有运行指令,运行时修改训练或者测试的路径
   我们将所有的训练结果整合放在了pkl-A和pkl-B文件夹下，便于最终融合时读取(pkl-A是对A数据集,pkl-B是对B数据集)
   对于B数据集的测试标签为自己做的标签，运行时自己生成一个B测试集的标签以便代码正常运行
   融合时注意数据路径
```


## 6. 源代码位置
- 6.1 (http://gj03.khdxs7.site:22903/QiranLi0823/UAV-1.git)
- 6.2 我们自己训练的权重在github链接上均可下载



注:模型参考和运行指令参考链接(https://github.com/happylinze/UAV-SAR)


### training

mixformer

1. We trained the model in k=1, k=2 and k=6. You could use the commands as:
```shell
cd ./mixformer

# k=1
python main.py --config ./config/uav_csv2/_1.yaml --work-dir <the save path of results>

# k=2
python main.py --config ./config/uav_csv2/_2.yaml --work-dir <the save path of results>

# k=6
python main.py --config ./config/uav_csv2/_6.yaml --work-dir <the save path of results>
```

2. We also trained the model in k=1, k=2 and k=6 with motion. The commands as:
```shell
cd ./mixformer

# By motion k=1
python main.py --config ./config/uav_csv2/motion_1.yaml --work-dir <the save path of results>

# By motion k=2
python main.py --config ./config/uav_csv2/motion_2.yaml --work-dir <the save path of results>

# By motion k=6
python main.py --config ./config/uav_csv2/motion_6.yaml --work-dir <the save path of results>
```

3. And we tried the angle data to train the model. The command as:
```shell
cd ./mixformer

# use angle
python main.py --config ./config/uav_csv2/angle_1.yaml --work-dir <the save path of results>

```


STTFormer

We trained joint, bone and motion. The commands as follows:
```shell
cd ./sttformer

# train joint
python main.py --config ./config/uav_csv2/joint.yaml --work-dir <the save path of results>

# train bone
python main.py --config ./config/uav_csv2/bone.yaml --work-dir <the save path of results>

# train motion
python main.py --config ./config/uav_csv2/motion.yaml --work-dir <the save path of results>

# train angle
python main.py --config ./config/uav_csv2/angle.yaml --work-dir <the save path of results>

```

### Testing
If you want to test any trained model saved in `<work_dir>`, run the following command: 
```shell
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save_score True --weights <work_dir>/xxx.pt
```

最终结果整合保存为形式为
```.
- pkl-A/
      - mix_1/
         - epoch1_test_score.pkl
      - mix_2/
         - epoch1_test_score.pkl
      - mix_6/
         - epoch1_test_score.pkl
      - mix_angle_1/
         - epoch1_test_score.pkl
      - mix_motion_1/
         - epoch1_test_score.pkl
      - mix_motion_2/
         - epoch1_test_score.pkl
      - mix_motion_6/
         - epoch1_test_score.pkl
      - ntu60_A
         - epoch1_test_score.pkl
      - ntu60_B
         - epoch1_test_score.pkl
      - ntu60_J
         - epoch1_test_score.pkl
      - ntu60_M
         - epoch1_test_score.pkl   
            ...
    - pkl-B/
      - mix_1/
         - epoch1_test_score.pkl
      - mix_2/
         - epoch1_test_score.pkl
      - mix_6/
         - epoch1_test_score.pkl
      - mix_angle_1/
         - epoch1_test_score.pkl
      - mix_motion_1/
         - epoch1_test_score.pkl
      - mix_motion_2/
         - epoch1_test_score.pkl
      - mix_motion_6/
         - epoch1_test_score.pkl
      - ntu60_A
         - epoch1_test_score.pkl
      - ntu60_B
         - epoch1_test_score.pkl
      - ntu60_J
         - epoch1_test_score.pkl
      - ntu60_M
         - epoch1_test_score.pkl
         ...
```
Then run the command as:
```shell
python ensemble.py
```

