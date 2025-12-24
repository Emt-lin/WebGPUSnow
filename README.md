## 起步

使用 WebGPU 实现的高性能下雪粒子效果插件，为 Typecho 博客添加冬日氛围。

## 特性

- 使用 WebGPU Compute Shader 实现高性能粒子系统
- 支持 Worker 线程隔离渲染，不阻塞主线程
- 自动降级：Worker WebGPU → 主线程 WebGPU → Canvas 2D
- 可配置粒子数量、速度、风力等参数

## 使用方法

第一步：下载本插件，放在 `usr/plugins/` 目录中；

第二步：文件夹名改为 WebGPUSnow；

第三步：登录管理后台，激活插件；

可以在后台设置粒子数量、下落速度、风力等参数。

## 浏览器支持

- Chrome 113+
- Edge 113+
- Safari (需开启 WebGPU 实验性功能)
- 不支持 WebGPU 的浏览器自动降级到 Canvas 2D

github开源地址：[WebGPUSnow][1]

## 与我联系：

作者：[未央花][2]
或者通过 mail: emt934841028@gmail.com
有任何问题也可评论留言

## 感谢

参考实现：[cyandev.app][3]

[1]: https://github.com/Emt-lin/WebGPUSnow
[2]: https://www.pslanys.com/
[3]: https://cyandev.app/post/make-a-snowfall-with-webgpu
