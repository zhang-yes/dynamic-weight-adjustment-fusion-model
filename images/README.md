# 可见光数据集

该数据集包含了810张由无人机拍摄的大豆照片，图片的描述信息，对应品种的含水量信息见`source.xlsx`文件夹

## 1 使用的传感器参数

**Table 1 Spectral channel parameters**

| Serial number | Band Name | Central wavelength/nm | Band width/nm | Standard gray board reflectance |
| :------------ | :-------- | :-------------------- | :------------ | :------------------------------ |
| 1             | Blue      | 450                   | 35            | 0.61                            |
| 2             | Green     | 555                   | 27            | 0.60                            |
| 3             | Red       | 660                   | 22            | 0.60                            |
| 4             | Re1       | 720                   | 10            | 0.61                            |
| 5             | Re2       | 750                   | 10            | 0.61                            |
| 6             | Nir       | 840                   | 30            | 0.60                            |

Note: Blue represents the blue band, Green represents the green band, Red represents the red band, Re1 represents the red edge 1 band, Re2 represents the red edge 2 band, and Nir represents the near infrared band.

 **Table 2 Specific parameters of RGB sensor**

| Technical index  | Specific parameters | Technical index             | Specific parameters  |
| :--------------- | :------------------ | :-------------------------- | :------------------- |
| photograph       | 35.9×24 mm          | ISO range                   | 100-25600            |
| Effective pixel  | 45 million          | Minimum photo interval      | 0.7 seconds          |
| Pixel size       | 4.4 μm              | Stable system               | Pitch, roll, pan     |
| Supported lens   | 35mm，FOV63.5°      | Angular jitter              | ±0.01°               |
| Memory card type | SD card             | Controllable rotation range | Pitch: -130° to +40° |
| Working mode     | Photo mode          | Roll: -55° to +55°          |                      |
| Aperture range   | f/2.8-f/16          | Translation: ±320°          |                      |

 

**Table 3 Specific parameters of multispectral sensors**

| Technical indicators | Specific parameters                  | Technical indicators | Specific parameters   |
| :------------------- | :----------------------------------- | :------------------- | :-------------------- |
| Target size          | 1/3"                                 | Main engine size     | ≤130 mm×160 mm×150 mm |
| Effective pixel      | 1.2Mpx                               | Picture format       | 16bit raw TIFF        |
| Quantization number  | 12bit                                | Shutter type         | Overall situation     |
| Field of view        | 49.6°×38°                            | Power supply         | X-Port                |
| Ground resolution    | [8.65 cm@h120](mailto:8.65cm@h120) m | Power dissipation    | 7 W/10 W              |
| Covering width       | 110 m×83 m@h120 m                    | Processing software  | Yusense Map           |

## 2 贡献

欢迎对项目进行贡献。请fork仓库并提交pull request以改进项目。

## 3 引用许可证
若使用本项目代码和数据，请在文章中标注或者链接项目地址 `https://github.com/zhang-yes/dynamic-weight-adjustment-fusion-model`

